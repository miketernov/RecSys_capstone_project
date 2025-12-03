// ======================================
// CONFIG — ПУТИ К МОДЕЛИ И ТОКЕНИЗАТОРУ
// ======================================

// Модель (model.onnx) лежит в GitHub Release
// !!! ЗАМЕНИ на свой URL !!!
const MODEL_URL =
  "https://github.com/miketernov/RecSys_capstone_project/releases/download/v1/model.onnx";

// Эти файлы лежат внутри репозитория в папке /model/
const TOKENIZER_URL = "model/tokenizer.json";

const TOTAL_CHUNKS = 17;

let recipes = [];
let tokenizer = null;
let session = null;

const loading = document.getElementById("loading");
const progress = document.getElementById("progress");
const resultsDiv = document.getElementById("results");


// ======================================
// LOAD RECIPE CHUNKS
// ======================================
async function loadChunks() {
    let all = [];
    for (let i = 1; i <= TOTAL_CHUNKS; i++) {
        progress.textContent = `Loading recipes ${i}/${TOTAL_CHUNKS}…`;
        const data = await fetch(`chunks/part${i}.json`).then(r => r.json());
        all.push(...data);
    }
    recipes = all;
    progress.textContent = "";
    console.log("Loaded recipes:", recipes.length);
}


// ======================================
// LOAD TOKENIZER
// ======================================
async function loadTokenizer() {
    tokenizer = await fetch(TOKENIZER_URL).then(r => r.json());
    console.log("Tokenizer loaded");
}


// ======================================
// LOAD ONNX MODEL
// ======================================
async function loadModel() {
    session = await ort.InferenceSession.create(MODEL_URL);
    console.log("ONNX model loaded");
}


// ======================================
// TOKENIZE TEXT → IDS
// ======================================
function tokenize(text) {
    const vocab = tokenizer.vocab;
    const cls = vocab["[CLS]"];
    const sep = vocab["[SEP]"];
    const unk = vocab["[UNK]"];

    let tokens = text.toLowerCase().split(/[^a-z]+/).filter(x => x);
    let ids = tokens.map(t => vocab[t] ?? unk);

    return [cls, ...ids, sep];
}


// ======================================
// RUN MODEL → GET EMBEDDING
// ======================================
async function embedText(text) {
    const ids = tokenize(text);
    const mask = ids.map(_ => 1);

    const input_ids = new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
    const attention_mask = new ort.Tensor("int64", BigInt64Array.from(mask.map(BigInt)), [1, mask.length]);

    const output = await session.run({
        input_ids,
        attention_mask
    });

    const hidden = output.last_hidden_state.data;
    const dim = hidden.length / ids.length;

    // mean pooling
    let res = new Array(dim).fill(0);
    for (let t = 0; t < ids.length; t++) {
        for (let d = 0; d < dim; d++) {
            res[d] += hidden[t * dim + d];
        }
    }
    return res.map(v => v / ids.length);
}


// ======================================
// COSINE SIMILARITY
// ======================================
function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}


// ======================================
// EXPLANATION
// ======================================
function explain(query, recipe) {
    const q = query.toLowerCase().split(/[^a-z]+/);
    const text = recipe.ingredients.join(" ").toLowerCase();

    const overlap = q.filter(w => text.includes(w));
    return overlap.length
        ? `Uses your ingredients: ${overlap.join(", ")}`
        : `Semantically similar to your ingredients.`;
}


// ======================================
// RECOMMEND
// ======================================
async function recommend() {
    const query = document.getElementById("ingredientsInput").value.trim();
    if (!query) return;

    loading.textContent = "Encoding ingredients with MiniLM…";

    const qvec = await embedText("ingredients: " + query);

    const scored = recipes.map(r => ({
        recipe: r,
        score: cosine(qvec, r.embedding)
    }));

    scored.sort((a, b) => b.score - a.score);

    const top = scored.slice(0, 3);

    resultsDiv.innerHTML = top
        .map(x => `
        <div class="recipe-card">
            <h3>${x.recipe.cuisine.toUpperCase()}</h3>
            <b>Score:</b> ${x.score.toFixed(4)}<br>
            <b>Ingredients:</b> ${x.recipe.ingredients.join(", ")}<br>
            <i>${explain(query, x.recipe)}</i>
        </div>`)
        .join("");

    loading.textContent = "";
}


// ======================================
// INIT
// ======================================
async function init() {
    loading.textContent = "Loading recipes…";
    await loadChunks();

    loading.textContent = "Loading tokenizer…";
    await loadTokenizer();

    loading.textContent = "Loading model (MiniLM)…";
    await loadModel();

    loading.textContent = "Ready ✔";
}

init();
document.getElementById("searchBtn").onclick = recommend;
