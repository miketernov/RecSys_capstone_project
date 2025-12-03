// ======================================
// CONFIG — МОДЕЛЬ И ТОКЕНИЗАТОР
// ======================================

// RAW ФАЙЛ model.onnx из твоего Releases (CORS-friendly):
const MODEL_URL =
  "https://github.com/miketernov/RecSys_capstone_project/releases/download/v1/model.onnx";

// Локальный токенизатор из /model/
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
        const part = await fetch(`chunks/part${i}.json`).then(r => r.json());
        all.push(...part);
    }
    recipes = all;
    progress.textContent = "";
    console.log("Recipes loaded:", recipes.length);
}


// ======================================
// LOAD TOKENIZER
// ======================================
async function loadTokenizer() {
    tokenizer = await fetch(TOKENIZER_URL).then(r => r.json());

    console.log("Tokenizer loaded");
    console.log("CLS token index:", tokenizer.vocab["[CLS]"]);
}


// ======================================
// LOAD MODEL
// ======================================
async function loadModel() {
    session = await ort.InferenceSession.create(MODEL_URL);
    console.log("MiniLM model loaded");
}


// ======================================
// TOKENIZATION
// ======================================
function tokenize(text) {
    // корректный путь к словарю
    const vocab = tokenizer.model.vocab;

    const CLS = vocab["[CLS]"];
    const SEP = vocab["[SEP]"];
    const UNK = vocab["[UNK]"];

    if (CLS === undefined || SEP === undefined || UNK === undefined) {
        console.error("Tokenizer is loaded, but CLS/SEP/UNK are not found");
        console.log("Available keys:", Object.keys(vocab).slice(0,20));
    }

    const tokens = text
        .toLowerCase()
        .split(/[^a-z]+/)
        .filter(t => t.length > 0);

    const ids = tokens.map(t => vocab[t] ?? UNK);

    return [CLS, ...ids, SEP];
}


// ======================================
// EMBEDDING USING ONNX MODEL
// ======================================
async function embedText(text) {
    const ids = tokenize(text);
    const mask = ids.map(_ => 1);

    const input_ids = new ort.Tensor(
        "int64",
        BigInt64Array.from(ids.map(BigInt)),
        [1, ids.length]
    );

    const attention_mask = new ort.Tensor(
        "int64",
        BigInt64Array.from(mask.map(BigInt)),
        [1, mask.length]
    );

    const output = await session.run({
        input_ids,
        attention_mask
    });

    const hidden = output.last_hidden_state.data;

    const dim = hidden.length / ids.length;
    let embedding = new Array(dim).fill(0);

    // mean-pooling
    for (let t = 0; t < ids.length; t++) {
        for (let d = 0; d < dim; d++) {
            embedding[d] += hidden[t * dim + d];
        }
    }

    return embedding.map(v => v / ids.length);
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
    const userWords = query.toLowerCase().split(/[^a-z]+/);
    const recipeText = recipe.ingredients.join(" ").toLowerCase();

    const overlap = userWords.filter(w => w && recipeText.includes(w));

    if (overlap.length > 0) {
        return `Contains your ingredients: ${overlap.join(", ")}`;
    }

    return "Semantically similar combination of ingredients.";
}


// ======================================
// MAIN SEARCH
// ======================================
async function recommend() {
    const query = document.getElementById("ingredientsInput").value.trim();
    if (!query) return;

    loading.textContent = "Encoding ingredients with MiniLM…";

    const queryVec = await embedText("ingredients: " + query);

    const scored = recipes.map((r, i) => ({
        recipe: r,
        score: cosine(queryVec, r.embedding)
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
        </div>
        `)
        .join("");

    loading.textContent = "";
}


// ======================================
// INIT PIPELINE
// ======================================
async function init() {
    loading.textContent = "Loading recipes…";
    await loadChunks();

    loading.textContent = "Loading tokenizer…";
    await loadTokenizer();

    loading.textContent = "Loading MiniLM model…";
    await loadModel();

    loading.textContent = "Ready ✔";
}

init();

document.getElementById("searchBtn").onclick = recommend;
