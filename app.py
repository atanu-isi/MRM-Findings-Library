from flask import Flask, render_template, request, jsonify
import json
import re
import math
from collections import defaultdict

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='/')

# ── In-memory store ───────────────────────────────────────────────────────────
findings_db = []
clusters_db = []

# ── Minimal TF-IDF vectorizer ─────────────────────────────────────────────────
# Minimal stop-word list — intentionally lean so domain terms survive.
# Removed generic MRM terms like "model", "risk", "finding" that were OVER-filtering.
STOP_WORDS = {
    'a','an','the','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','will','would','could','should','may','might',
    'shall','must','can','need','dare','ought','used','to','of','in','for',
    'on','with','at','by','from','as','into','through','during','before',
    'after','above','below','between','out','off','over','under','again',
    'further','then','once','and','but','or','nor','so','yet','both',
    'either','neither','not','only','own','same','than','too','very','just',
    'because','if','while','although','though','since','unless','until',
    'that','this','these','those','it','its','such','no','each','every',
    'all','any','few','more','most','other','some','what','which',
    'who','whom','there','their','they','we','our','us','you','your','he',
    'she','him','her','his','i','me','my','also','however','therefore',
    'thus','hence','moreover','furthermore','additionally','based','upon',
    'lead','leads','result','results','increase','increases','lack','lacks',
    'absence','without','ensure','including','across','within','where',
    'when','how','well','due','per','its','has','been',
}

# ── Domain-aware field weights ────────────────────────────────────────────────
# model_theme is the single strongest signal — weight it heavily.
# Title is also highly discriminative. Description and BJ provide supporting context.
FIELD_WEIGHTS = {
    'model_theme': 6,
    'title': 4,
    'description': 2,
    'business_justification': 1,
}

# ── Known MRM theme → canonical group mapping ─────────────────────────────────
# This lets us seed clustering with structural knowledge when a theme is present.
THEME_GROUPS = {
    'modelling input data':  'data_quality',
    'modeling input data':   'data_quality',
    'model input data':      'data_quality',
    'input data':            'data_quality',
    'model documentation':   'documentation',
    'documentation':         'documentation',
    'model governance':      'governance',
    'governance':            'governance',
    'model methodology':     'methodology',
    'methodology':           'methodology',
    'model performance':     'performance_monitoring',
    'performance':           'performance_monitoring',
    'model implementation':  'performance_monitoring',
    'implementation':        'performance_monitoring',
}

def normalize_theme(theme: str) -> str:
    if not theme:
        return ''
    return THEME_GROUPS.get(theme.lower().strip(), theme.lower().strip())


def tokenize(text: str) -> list:
    text = text.lower()
    tokens = re.findall(r'\b[a-z][a-z\-]{2,}\b', text)
    return [t for t in tokens if t not in STOP_WORDS]


def build_weighted_doc(finding: dict) -> str:
    """
    Build a composite document string that repeats each field
    in proportion to its weight, giving heavier fields more
    influence on the final TF-IDF vector.
    """
    parts = []
    for field, weight in FIELD_WEIGHTS.items():
        val = finding.get(field, '') or ''
        # Normalise model_theme to canonical group before repeating
        if field == 'model_theme':
            val = normalize_theme(val)
        parts.extend([val] * weight)
    return ' '.join(parts)


def build_tfidf(docs: list):
    N = len(docs)
    tokenized = [tokenize(d) for d in docs]
    df = defaultdict(int)
    for toks in tokenized:
        for t in set(toks):
            df[t] += 1
    vocab = list(df.keys())
    word2idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    vectors = []
    for toks in tokenized:
        tf = defaultdict(int)
        for t in toks:
            tf[t] += 1
        vec = [0.0] * V
        for t, cnt in tf.items():
            if t in word2idx:
                idf = math.log((N + 1) / (df[t] + 1)) + 1
                vec[word2idx[t]] = (cnt / len(toks)) * idf if toks else 0
        norm = math.sqrt(sum(x * x for x in vec)) or 1
        vectors.append([x / norm for x in vec])
    return vectors, vocab, word2idx


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def kmeans(vectors, k, max_iter=300, n_restarts=5, seed=42):
    """
    K-Means with k-means++ initialisation and multiple restarts.
    Returns the labelling with the best within-cluster inertia.
    """
    import random
    n = len(vectors)
    if n == 0:
        return [], []
    k = min(k, n)

    best_labels = None
    best_inertia = float('inf')
    best_centroids = None

    for restart in range(n_restarts):
        random.seed(seed + restart * 31)

        # k-means++ initialisation
        centroids = [vectors[random.randint(0, n - 1)][:]]
        for _ in range(k - 1):
            dists = []
            for v in vectors:
                d = min(1 - cosine(v, c) for c in centroids)
                dists.append(max(d, 0))
            total = sum(dists) or 1
            r = random.random() * total
            cum = 0
            chosen = vectors[-1][:]
            for i, d in enumerate(dists):
                cum += d
                if cum >= r:
                    chosen = vectors[i][:]
                    break
            centroids.append(chosen)

        labels = [0] * n
        for _ in range(max_iter):
            new_labels = []
            for v in vectors:
                sims = [cosine(v, c) for c in centroids]
                new_labels.append(sims.index(max(sims)))
            if new_labels == labels:
                break
            labels = new_labels
            # Recompute centroids; reinitialise empty clusters
            for j in range(k):
                members = [vectors[i] for i, l in enumerate(labels) if l == j]
                if members:
                    centroids[j] = [sum(col) / len(members) for col in zip(*members)]
                else:
                    # Empty cluster: steal the point farthest from its centroid
                    farthest = max(range(n), key=lambda i: 1 - cosine(vectors[i], centroids[labels[i]]))
                    centroids[j] = vectors[farthest][:]

        # Compute inertia (sum of distances to assigned centroid)
        inertia = sum(1 - cosine(vectors[i], centroids[labels[i]]) for i in range(n))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels[:]
            best_centroids = [c[:] for c in centroids]

    return best_labels, best_centroids


def _silhouette_score(vectors, labels, k):
    """
    Compute the mean silhouette coefficient for a clustering solution.
    Returns a value in [-1, 1]; higher is better.
    """
    n = len(vectors)
    if k <= 1 or n <= k:
        return -1.0

    # Group vectors by cluster
    clusters = defaultdict(list)
    for i, l in enumerate(labels):
        clusters[l].append(i)

    scores = []
    for i, label in enumerate(labels):
        same = [j for j in clusters[label] if j != i]
        if not same:
            scores.append(0.0)
            continue
        # Mean intra-cluster distance
        a = sum(1 - cosine(vectors[i], vectors[j]) for j in same) / len(same)
        # Mean nearest-cluster distance
        b_vals = []
        for other_label, members in clusters.items():
            if other_label == label:
                continue
            mean_dist = sum(1 - cosine(vectors[i], vectors[j]) for j in members) / len(members)
            b_vals.append(mean_dist)
        b = min(b_vals) if b_vals else 0.0
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)

    return sum(scores) / len(scores) if scores else -1.0


def _auto_select_k(vectors, n: int, has_themes: bool, theme_count: int) -> int:
    """
    Automatically determine the optimal number of clusters using the
    silhouette method. Searches over a sensible range for the dataset size
    and picks the k with the highest mean silhouette score.

    Falls back gracefully for very small datasets.
    """
    if n <= 2:
        return 1
    if n == 3:
        return 2

    # Define search range based on dataset size
    if n <= 10:
        k_min, k_max = 2, max(2, n - 1)
    elif n <= 50:
        k_min, k_max = 2, min(15, n // 2)
    elif n <= 200:
        k_min, k_max = 2, min(30, n // 5)
    else:
        k_min, k_max = 2, min(60, n // 10)

    # If themes exist, also include theme_count in the search range as a candidate
    if has_themes and theme_count >= k_min:
        k_max = max(k_max, min(theme_count + 3, n - 1))

    best_k = k_min
    best_score = -2.0

    for k in range(k_min, k_max + 1):
        labels, _ = kmeans(vectors, k)
        if not labels:
            continue
        score = _silhouette_score(vectors, labels, k)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def _estimate_k(vectors, n: int, has_themes: bool) -> int:
    """
    Determine optimal k automatically using silhouette scoring.
    - Counts distinct canonical theme groups as a hint for the search range.
    - Runs silhouette analysis across the plausible k range.
    - Returns the k that maximises cluster cohesion/separation.
    """
    theme_count = 0
    if has_themes:
        themes = set()
        for f in findings_db:
            t = normalize_theme(f.get('model_theme', ''))
            if t:
                themes.add(t)
        theme_count = len(themes)

    return _auto_select_k(vectors, n, has_themes, theme_count)


def _top_signals(fids):
    texts = []
    for fid in fids:
        f = next((x for x in findings_db if x['finding_id'] == fid), None)
        if f:
            texts.append(build_weighted_doc(f))
    tokens = []
    for t in texts:
        tokens.extend(tokenize(t))
    freq = defaultdict(int)
    for t in tokens:
        freq[t] += 1
    top = sorted(freq, key=lambda x: -freq[x])[:10]
    return ', '.join(top)


def _generate_why(fids):
    themes = []
    for fid in fids:
        f = next((x for x in findings_db if x['finding_id'] == fid), None)
        if f and f.get('model_theme'):
            themes.append(f['model_theme'])
    theme_str = ''
    if themes:
        unique = list(dict.fromkeys(themes))
        theme_str = f" under the theme(s): {', '.join(unique)}."
    return (
        f"Findings {', '.join(fids)} share overlapping concepts in their title, "
        f"description, and business justifications{theme_str} "
        "Semantic clustering detected strong similarity in vocabulary, "
        "risk focus, and remediation context across all relevant fields."
    )


def run_clustering(k: int = None):
    global clusters_db
    if not findings_db:
        clusters_db = []
        return

    # Determine whether model_theme is available
    has_themes = any(f.get('model_theme') for f in findings_db)

    # Build weighted composite documents using all fields
    docs = [build_weighted_doc(f) for f in findings_db]

    vectors, vocab, word2idx = build_tfidf(docs)
    n = len(findings_db)

    if k is None:
        k = _estimate_k(vectors, n, has_themes)
    k = min(k, n)

    labels, centroids = kmeans(vectors, k)

    cluster_map = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_map[label].append(findings_db[i]['finding_id'])

    # ── Optional: sort clusters by dominant model_theme for stable labelling ──
    def cluster_theme_sort_key(fids):
        theme_counts = defaultdict(int)
        for fid in fids:
            f = next((x for x in findings_db if x['finding_id'] == fid), None)
            if f:
                theme_counts[normalize_theme(f.get('model_theme', ''))] += 1
        dominant = max(theme_counts, key=theme_counts.get) if theme_counts else ''
        theme_order = ['data_quality', 'documentation', 'governance', 'methodology', 'performance_monitoring']
        try:
            return theme_order.index(dominant)
        except ValueError:
            return 99

    sorted_clusters = sorted(cluster_map.values(), key=cluster_theme_sort_key)

    clusters_db = []
    for idx, fids in enumerate(sorted_clusters):
        clusters_db.append({
            'cluster_id': f"C{idx + 1}",
            'findings_included': fids,
            'why_grouped': _generate_why(fids),
            'semantic_signals': _top_signals(fids),
            'size': len(fids),
        })


def search_query(query: str) -> dict:
    if not clusters_db or not findings_db:
        return {'cluster': None, 'findings': [], 'score': 0}
    docs = [build_weighted_doc(f) for f in findings_db]
    all_docs = docs + [query]
    vectors, vocab, word2idx = build_tfidf(all_docs)
    q_vec = vectors[-1]
    doc_vecs = vectors[:-1]
    sims = [(cosine(q_vec, dv), i) for i, dv in enumerate(doc_vecs)]
    sims.sort(reverse=True)
    top_findings = [findings_db[i] for _, i in sims[:5]]
    top_ids = {f['finding_id'] for f in top_findings}
    best_cluster = None
    best_count = 0
    for c in clusters_db:
        overlap = len(set(c['findings_included']) & top_ids)
        if overlap > best_count:
            best_count = overlap
            best_cluster = c
    return {
        'cluster': best_cluster,
        'findings': top_findings,
        'score': round(sims[0][0], 3) if sims else 0,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/findings', methods=['GET'])
def get_findings():
    return jsonify(findings_db)


@app.route('/api/findings', methods=['POST'])
def add_finding():
    data = request.json
    required = ['finding_id', 'model_theme','title', 'description', 'business_justification']
    if not all(k in data for k in required):
        return jsonify({'error': 'Missing required fields: finding_id, title, description, business_justification'}), 400
    if any(f['finding_id'] == data['finding_id'] for f in findings_db):
        return jsonify({'error': 'Finding ID already exists'}), 409
    record = {k: str(data[k]).strip() for k in required}
    if 'model_theme' in data and data['model_theme']:
        record['model_theme'] = str(data['model_theme']).strip()
    findings_db.append(record)
    run_clustering()
    return jsonify({'status': 'ok', 'total': len(findings_db)}), 201


@app.route('/api/findings/bulk', methods=['POST'])
def bulk_findings():
    rows = request.json
    added, skipped = 0, 0
    for row in rows:
        required = ['finding_id','model_theme','title', 'description', 'business_justification']
        if not all(k in row for k in required):
            skipped += 1
            continue
        if any(f['finding_id'] == row['finding_id'] for f in findings_db):
            skipped += 1
            continue
        record = {k: str(row[k]).strip() for k in required}
        if 'model_theme' in row and row['model_theme']:
            record['model_theme'] = str(row['model_theme']).strip()
        findings_db.append(record)
        added += 1
    run_clustering()
    return jsonify({'added': added, 'skipped': skipped, 'total': len(findings_db)})


@app.route('/api/findings/<fid>', methods=['DELETE'])
def delete_finding(fid):
    global findings_db
    findings_db = [f for f in findings_db if f['finding_id'] != fid]
    run_clustering()
    return jsonify({'status': 'ok'})


@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    return jsonify(clusters_db)


@app.route('/api/clusters/rerun', methods=['POST'])
def rerun_clustering():
    k = request.json.get('k') if request.json else None
    run_clustering(k)
    return jsonify({'status': 'ok', 'clusters': len(clusters_db)})


@app.route('/api/search', methods=['POST'])
def search():
    query = (request.json or {}).get('query', '')
    if not query:
        return jsonify({'error': 'Empty query'}), 400
    result = search_query(query)
    return jsonify(result)


@app.route('/api/stats', methods=['GET'])
def stats():
    return jsonify({
        'total_findings': len(findings_db),
        'total_clusters': len(clusters_db),
    })


# ── Seed with sample data on first run ───────────────────────────────────────
SAMPLE_FINDINGS = [
    {"finding_id": "F001", "model_theme": "Modelling Input Data",
     "title": "Inadequate Data Quality Controls for Model Input Data",
     "description": "The model relies on multiple upstream source systems (including loan origination, customer master, and external bureau data); however, there is no formalized and consistently applied data quality control framework governing the ingestion and preprocessing stages. Data validation checks (e.g., missing value thresholds, outlier detection, referential integrity checks) are either absent or implemented inconsistently across different data feeds. Additionally, there is no automated reconciliation between source systems and model input datasets to ensure completeness. Data transformations applied during preprocessing are not fully documented, and there is limited evidence of controls to detect data corruption or unintended alterations during data handling. Historical data used for model development also shows gaps in key fields, with no documented imputation strategy or justification.",
     "business_justification": "Deficiencies in data quality controls increase the risk that the model is trained and executed on inaccurate, incomplete, or inconsistent data, directly affecting the reliability of model outputs. This can lead to systematic bias in risk estimates, misinformed credit decisions, and potential misstatement of risk-weighted assets. Lack of robust data governance frameworks may result in non-compliance with supervisory expectations around data integrity, increasing the likelihood of regulatory findings or capital penalties."},
    {"finding_id": "F002", "model_theme": "Model Documentation",
     "title": "Insufficient Model Development Documentation and Transparency",
     "description": "The model documentation does not provide a comprehensive and traceable account of the development process. Key elements such as variable selection criteria, feature engineering steps, transformation logic, and exclusion of candidate variables are not adequately described. The rationale for selecting the final model methodology over alternative approaches is not documented, and there is limited discussion of model limitations and assumptions. Furthermore, the documentation lacks reproducibility — there is no clear linkage between documented steps and the actual codebase or datasets used.",
     "business_justification": "Inadequate documentation reduces transparency and impairs the ability of independent validation teams and auditors to effectively review and challenge the model. This increases model risk by allowing potential conceptual or technical weaknesses to remain unidentified. Additionally, poor documentation creates key-person dependency, as future redevelopment or recalibration efforts may not be feasible without significant rework. Regulatory expectations emphasize comprehensive documentation as a cornerstone of model risk management."},
    {"finding_id": "F003", "model_theme": "Model Governance",
     "title": "Unclear Model Ownership and Weak Governance Structure",
     "description": "The model governance framework does not clearly define roles and responsibilities across model development, validation, approval, and ongoing monitoring functions. There is no formally designated model owner accountable for the model's performance and compliance throughout its lifecycle. The model inventory is incomplete and does not consistently capture key attributes such as model tiering, usage, approval status, and review frequency. Escalation procedures for model-related issues are not formally documented or consistently followed.",
     "business_justification": "Weak governance structures create ambiguity in accountability, increasing the risk that model deficiencies remain unaddressed or are not escalated in a timely manner. This may lead to continued reliance on flawed models for critical business decisions. Lack of governance oversight may result in non-compliance with internal policies and regulatory expectations, potentially triggering supervisory findings and enforcement actions."},
    {"finding_id": "F004", "model_theme": "Model Methodology",
     "title": "Unjustified and Untested Model Assumptions",
     "description": "The model incorporates several key assumptions, including linear relationships between predictors and target variables, independence among explanatory variables, and stationarity of underlying data distributions. However, these assumptions have not been empirically tested or validated through statistical analysis. There is no evidence of diagnostic testing (e.g., multicollinearity checks, residual analysis, stability tests) to support the appropriateness of the chosen methodology. Alternative modeling approaches that may better capture non-linear relationships or interactions were not considered.",
     "business_justification": "Failure to validate core model assumptions undermines the conceptual soundness of the model and increases the likelihood of biased or unstable outputs. This can result in incorrect risk assessments, particularly under changing economic conditions or portfolio characteristics. Models lacking methodological rigor may be deemed unreliable by regulators, leading to increased scrutiny or disqualification for regulatory capital purposes."},
    {"finding_id": "F005", "model_theme": "Model Performance",
     "title": "Absence of Robust Ongoing Performance Monitoring Framework",
     "description": "Post-implementation monitoring of model performance is not supported by a formalized framework. Key performance indicators such as discriminatory power (e.g., Gini coefficient), calibration accuracy, population stability index (PSI), and characteristic stability index (CSI) are either not defined or not monitored on a periodic basis. There are no established thresholds or trigger limits to identify model deterioration. Monitoring activities, where performed, are manual and lack proper documentation, limiting traceability.",
     "business_justification": "Without systematic performance monitoring, degradation in model accuracy or stability may go undetected, leading to continued reliance on outdated or mis-specified models. This can adversely impact credit approvals, pricing decisions, and may result in financial losses. Regulators expect institutions to continuously monitor model performance and take timely corrective actions; failure increases the risk of supervisory intervention."},
    {"finding_id": "F006", "model_theme": "Model Implementation",
     "title": "Lack of Controls Ensuring Consistency Between Development and Production Environments",
     "description": "There are material discrepancies between the model implementation in the development environment and the version deployed in production. Differences were observed in data preprocessing logic, parameter values, and treatment of missing values. There is no formal reconciliation or validation process to ensure that the production implementation faithfully reflects the approved model. Deployment processes lack automated testing, version control, and segregation of duties, increasing the risk of unauthorized or erroneous changes.",
     "business_justification": "Implementation inconsistencies can lead to divergence between expected and actual model outputs, undermining the reliability of model-driven decisions. This introduces operational risk and may result in incorrect risk assessments or financial reporting. Lack of robust implementation controls weakens auditability and increases the likelihood of regulatory findings."},
    {"finding_id": "F007", "model_theme": "Modelling Input Data",
     "title": "Use of Non-Representative and Outdated Training Data",
     "description": "The model has been developed using historical data that does not adequately reflect the current portfolio composition, customer behavior, or prevailing macroeconomic conditions. There is no evidence of data refresh or recalibration since initial development. Additionally, structural breaks (e.g., post-pandemic changes in borrower behavior) have not been incorporated into the dataset. Sampling techniques used during model development are not documented, raising concerns about potential selection bias.",
     "business_justification": "Use of non-representative data can significantly impair model accuracy and lead to biased predictions, particularly in dynamic environments. This may result in underestimation or overestimation of risk, affecting capital adequacy and strategic decision-making. Regulators expect models to be based on relevant and representative data; failure may lead to model rejection or capital penalties."},
    {"finding_id": "F008", "model_theme": "Model Documentation",
     "title": "Incomplete Documentation of Model Limitations and Use Constraints",
     "description": "The model documentation does not clearly articulate the known limitations, assumptions, and appropriate use cases of the model. There is no guidance on scenarios where the model may produce unreliable results or should not be used. Additionally, there is no documentation of compensating controls or overlays required when limitations are triggered.",
     "business_justification": "Lack of clarity on model limitations increases the risk of misuse or over-reliance in inappropriate contexts. This can lead to incorrect decisions and potential financial losses. Proper documentation of limitations is a key regulatory expectation to ensure responsible model usage."},
    {"finding_id": "F009", "model_theme": "Model Governance",
     "title": "Inadequate Model Lifecycle Management and Review Process",
     "description": "The institution does not maintain a structured model lifecycle management framework covering development, validation, approval, implementation, monitoring, and decommissioning stages. Periodic model reviews are not conducted consistently, and there is no defined frequency based on model risk tiering.",
     "business_justification": "Weak lifecycle management increases the likelihood that outdated or high-risk models remain in use without proper oversight, leading to elevated model risk and potential regulatory non-compliance."},
    {"finding_id": "F010", "model_theme": "Model Methodology",
     "title": "Limited Consideration of Alternative Modeling Approaches",
     "description": "The model development process did not include a comprehensive benchmarking exercise against alternative methodologies. There is no evidence that more advanced or appropriate techniques were evaluated before finalizing the model.",
     "business_justification": "Failure to consider alternative approaches may result in suboptimal model performance and missed opportunities to improve predictive accuracy, ultimately impacting business outcomes and regulatory standing."},
    {"finding_id": "F011", "model_theme": "Model Performance",
     "title": "Lack of Backtesting and Benchmarking Analysis",
     "description": "The model has not been subjected to rigorous backtesting using out-of-sample data or benchmarking against challenger models. There is insufficient evidence of discriminatory power or calibration assessments on holdout datasets.",
     "business_justification": "Absence of backtesting limits confidence in the model's predictive power and stability, increasing the risk of relying on an underperforming model for critical business and regulatory decisions."},
    {"finding_id": "F012", "model_theme": "Model Implementation",
     "title": "Absence of Robust Version Control and Change Tracking",
     "description": "Model code and associated artifacts are not maintained in a controlled versioning system. Changes are not systematically logged, reviewed, or approved. There is no audit trail of modifications to model logic or parameters.",
     "business_justification": "Lack of version control increases operational risk and reduces traceability, making it difficult to investigate issues or demonstrate compliance during audits and regulatory reviews."},
]

for f in SAMPLE_FINDINGS:
    findings_db.append(f)
run_clustering()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
