// =====================================
// CONFIG
// =====================================
const TOTAL_CHUNKS = 17;
const ONNX_URL = "https://huggingface.co/iammik3e/recsys-minilm/resolve/main/model.onnx";
let recipes = [];
let session = null;
let tokenizer = null;

const loading = document.getElementById("loading");
const progress = document.getElementById("progress");

// =====================================
// IMPORT ONNX RUNTIME
// =====================================
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";

// =====================================
// LOAD TOKENIZER FILES
// =====================================
async function loadTokenizer() {
    const vocabUrl = "model/vocab.txt";
    const text = await fetch(vocabUrl).then(r => r.text());
    const lines = text.split("\n");

    const vocab = {};
    lines.forEach((t, i) => vocab[t.trim()] = i);

    tokenizer = { vocab };

    console.log("Tokenizer loaded:", Object.keys(vocab).length, "tokens");
}

// Encode text into input_ids (VERY SIMPLIFIED tokenizer)
function tokenize(text) {
    text = text.toLowerCase().replace(/[^a-z0-9 ]+/g, " ");
    const tokens = text.split(" ").filter(t => t.length > 0);

    const ids = tokens.map(t => tokenizer.vocab[t] ?? tokenizer.vocab["[UNK]"]);
    return ids.slice(0, 128);
}

// Convert ids→tensor for ONNX
function makeTensor(ids) {
    const input = new Int32Array(128);
    ids.forEach((v, i) => input[i] = v);
    return new ort.Tensor("int32", input, [1, 128]);
}

// =====================================
// LOAD MODEL
// =====================================
async function loadOnnxModel() {
    console.log("Loading MiniLM ONNX…");

    session = await ort.InferenceSession.create(ONNX_URL, {
        executionProviders: ["wasm"]
    });

    console.log("MiniLM loaded ✔");
}

// =====================================
// LOAD RECIPE CHUNKS
// =====================================
async function loadChunks() {
    if (recipes.length > 0) return;

    let all = [];
    for (let i = 1; i <= TOTAL_CHUNKS; i++) {
        progress.textContent = `Loading recipes ${i}/${TOTAL_CHUNKS}…`;
        const chunk = await fetch(`chunks/part${i}.json`).then(r => r.json());
        all.push(...chunk);
    }

    recipes = all;
    progress.textContent = "";
    console.log("Loaded recipes:", recipes.length);
}

// =====================================
// ENCODING USER INGREDIENTS
// =====================================
async function embed(text) {
    const ids = tokenize(text);
    const input_ids = makeTensor(ids);

    const outputs = await session.run({ input_ids });
    const emb = outputs.last_hidden_state.data;

    return emb;
}

// =====================================
// COSINE SIMILARITY
// =====================================
function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;

    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// =====================================
// MAIN SEARCH
// =====================================
async function recommend() {
    const txt = document.getElementById("ingredientsInput").value.trim();
    if (!txt) return;

    loading.textContent = "Encoding ingredients with MiniLM…";

    const userEmb = await embed(txt);

    const scores = recipes.map(r => ({
        recipe: r,
        score: cosine(userEmb, r.embedding)
    }));

    const top = scores.sort((a, b) => b.score - a.score).slice(0, 3);

    document.getElementById("results").innerHTML =
        top.map(x => `
        <div class="recipe-card">
            <h3>${x.recipe.cuisine.toUpperCase()}</h3>
            <b>Score:</b> ${x.score.toFixed(4)}<br>
            <b>Ingredients:</b> ${x.recipe.ingredients.join(", ")}<br><br>
            <i>This recipe matches your ingredients semantically, using MiniLM embedding similarity.</i>
        </div>
    `).join("");

    loading.textContent = "";
}

// =====================================
// INIT PIPELINE
// =====================================
async function init() {
    loading.textContent = "Loading recipes…";
    await loadChunks();

    loading.textContent = "Loading tokenizer…";
    await loadTokenizer();

    loading.textContent = "Loading MiniLM model (first time is slow)…";
    await loadOnnxModel();

    loading.textContent = "Ready ✔";
}

init();

document.getElementById("searchBtn").onclick = recommend;
