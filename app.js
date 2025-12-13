// ================= CONFIG =================
const ONNX_URL = "https://huggingface.co/iammik3e/recsys-minilm/resolve/main/model.onnx";
const RECIPES_FILE = "recipes_all.json";

let recipes = [];
let session = null;
let tokenizer = null;
let ort = null;

const loading = document.getElementById("loading");
const progress = document.getElementById("progress");

// ================= LOAD ONNX =================
async function loadONNXRuntime() {
    const ortModule = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.wasm.min.js");
    ort = ortModule.default || ortModule;
}

// ================= TOKENIZER =================
async function loadTokenizer() {
    const vocabText = await fetch("model/vocab.txt").then(r => r.text());
    const vocab = {};
    vocabText.split("\n").forEach((t, i) => vocab[t.trim()] = i);
    tokenizer = { vocab };
}

function tokenize(text) {
    text = text.toLowerCase().replace(/[^a-z0-9 ]+/g, " ");
    const tokens = text.split(" ").filter(Boolean);
    const ids = tokens.map(t => tokenizer.vocab[t] ?? tokenizer.vocab["[UNK]"]);
    const cls = tokenizer.vocab["[CLS]"] ?? 0;
    return { ids: [cls, ...ids].slice(0, 128), len: Math.min(ids.length + 1, 128) };
}

function makeTensor(ids) {
    const arr = new BigInt64Array(128);
    ids.forEach((v, i) => arr[i] = BigInt(v));
    return new ort.Tensor("int64", arr, [1, 128]);
}

// ================= LOAD MODEL =================
async function loadModel() {
    await loadONNXRuntime();
    session = await ort.InferenceSession.create(ONNX_URL, { executionProviders: ["wasm"] });
}

// ================= EMBEDDING =================
async function embed(text) {
    const tok = tokenize("Ingredients: " + text);
    const input_ids = makeTensor(tok.ids);

    const maskArr = new BigInt64Array(128);
    for (let i = 0; i < tok.len; i++) maskArr[i] = 1n;

    const outputs = await session.run({
        input_ids,
        attention_mask: new ort.Tensor("int64", maskArr, [1, 128]),
        token_type_ids: new ort.Tensor("int64", new BigInt64Array(128), [1, 128])
    });

    const data = outputs[Object.keys(outputs)[0]].data;
    const hidden = 384;
    const emb = new Array(hidden).fill(0);

    for (let i = 0; i < hidden; i++) {
        for (let j = 0; j < tok.len; j++) {
            emb[i] += data[j * hidden + i];
        }
        emb[i] /= tok.len;
    }
    return emb;
}

// ================= UTILS =================
function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function cuisineImage(cuisine) {
    return `https://source.unsplash.com/400x300/?${cuisine},food`;
}

// ================= SEARCH =================
async function recommend() {
    const query = ingredientsInput.value.trim();
    if (!query) return;

    const excluded = excludeInput.value
        .toLowerCase()
        .split(",")
        .map(x => x.trim())
        .filter(Boolean);

    const topN = parseInt(topNSelect.value);

    loading.textContent = "Encoding ingredients…";
    const userEmb = await embed(query);

    loading.textContent = "Searching…";

    const filtered = recipes.filter(r =>
        !excluded.some(e =>
            r.ingredients.join(" ").toLowerCase().includes(e)
        )
    );

    const scored = filtered.map(r => ({
        recipe: r,
        score: cosine(userEmb, r.embedding)
    }));

    const top = scored
        .sort((a, b) => b.score - a.score)
        .slice(0, topN);

    results.innerHTML = top.map(x => `
        <div class="recipe-card">
            <img src="${cuisineImage(x.recipe.cuisine)}">
            <h3>${x.recipe.cuisine.toUpperCase()}</h3>
            <b>Score:</b> ${x.score.toFixed(4)}<br>
            <b>Ingredients:</b> ${x.recipe.ingredients.join(", ")}
        </div>
    `).join("");

    loading.textContent = "";
}

// ================= INIT =================
async function init() {
    loading.textContent = "Loading…";
    await loadTokenizer();
    await loadModel();
    recipes = await fetch(RECIPES_FILE).then(r => r.json());
    loading.textContent = "Ready ✔";
}

const ingredientsInput = document.getElementById("ingredientsInput");
const excludeInput = document.getElementById("excludeInput");
const topNSelect = document.getElementById("topN");
const results = document.getElementById("results");

document.getElementById("searchBtn").onclick = recommend;

init();
