// =====================================
// CONFIG
// =====================================
const TOTAL_CHUNKS = 17;
const USE_SINGLE_FILE = true;
const RECIPES_FILE = "recipes_all.json";
const ONNX_URL = "https://huggingface.co/iammik3e/recsys-minilm/resolve/main/model.onnx";

let recipes = [];
let session = null;
let tokenizer = null;
let ort = null;

// UI
const loading = document.getElementById("loading");
const progress = document.getElementById("progress");
const results = document.getElementById("results");

// =====================================
// LOAD ONNX RUNTIME (ORIGINAL SAFE WAY)
// =====================================
async function loadONNXRuntime() {
    try {
        const ortModule = await import(
            "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.wasm.min.js"
        );
        ort = ortModule.default || ortModule;
        if (ort?.InferenceSession) {
            console.log("ONNX Runtime loaded via ES module ✔");
            return ort;
        }
    } catch (e) {
        console.warn("ES module load failed, fallback to script");
    }

    return new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.wasm.min.js";
        script.onload = () => {
            if (window.ort) {
                ort = window.ort;
                console.log("ONNX Runtime loaded via script ✔");
                resolve(ort);
            } else {
                reject(new Error("ORT not found"));
            }
        };
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

// =====================================
// TOKENIZER
// =====================================
async function loadTokenizer() {
    const text = await fetch("model/vocab.txt").then(r => r.text());
    const vocab = {};
    text.split("\n").forEach((t, i) => vocab[t.trim()] = i);
    tokenizer = { vocab };
}

function tokenize(text) {
    text = text.toLowerCase().replace(/[^a-z0-9 ]+/g, " ");
    const tokens = text.split(" ").filter(Boolean);
    const ids = tokens.map(t => tokenizer.vocab[t] ?? tokenizer.vocab["[UNK]"]);
    const cls = tokenizer.vocab["[CLS]"] ?? 0;
    const final = [cls, ...ids].slice(0, 128);
    return { ids: final, len: final.length };
}

function makeTensor(ids) {
    const arr = new BigInt64Array(128);
    ids.forEach((v, i) => arr[i] = BigInt(v));
    return new ort.Tensor("int64", arr, [1, 128]);
}

// =====================================
// LOAD MODEL
// =====================================
async function loadOnnxModel() {
    if (!ort) await loadONNXRuntime();

    if (!ort.InferenceSession && ort.default?.InferenceSession) {
        ort = ort.default;
    }

    if (!ort.InferenceSession) {
        throw new Error("InferenceSession not available");
    }

    session = await ort.InferenceSession.create(ONNX_URL, {
        executionProviders: ["wasm"]
    });

    console.log("MiniLM model loaded ✔");
}

// =====================================
// LOAD RECIPES (CHUNKS + CACHE)
// =====================================
async function loadChunks() {
    if (recipes.length) return;

    if (USE_SINGLE_FILE) {
        try {
            recipes = await fetch(RECIPES_FILE).then(r => r.json());
            console.log("Loaded recipes from single file ✔");
            return;
        } catch {
            console.warn("Single file failed, fallback to chunks");
        }
    }

    progress.textContent = "Loading recipe chunks…";
    const promises = [];

    for (let i = 1; i <= TOTAL_CHUNKS; i++) {
        promises.push(
            fetch(`chunks/part${i}.json`)
                .then(r => r.json())
                .catch(() => [])
        );
    }

    const chunks = await Promise.all(promises);
    recipes = chunks.flat();
    progress.textContent = "";

    console.log("Loaded recipes:", recipes.length);
}

// =====================================
// EMBEDDING
// =====================================
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

    const output = outputs[Object.keys(outputs)[0]];
    const data = output.data;
    const hidden = output.dims[2];

    const emb = new Array(hidden).fill(0);
    for (let i = 0; i < hidden; i++) {
        for (let j = 0; j < tok.len; j++) {
            emb[i] += data[j * hidden + i];
        }
        emb[i] /= tok.len;
    }
    return emb;
}

// =====================================
// SIMILARITY
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

function cuisineImage(cuisine) {
    return `https://source.unsplash.com/400x300/?${cuisine},food`;
}

// =====================================
// MAIN SEARCH (EXCLUSIONS + TOP-N + IMAGES)
// =====================================
async function recommend() {
    const query = document.getElementById("ingredientsInput").value.trim();
    const excludeText = document.getElementById("excludeInput").value;
    const topN = parseInt(document.getElementById("topN").value);

    if (!query) return;

    const excluded = excludeText
        .toLowerCase()
        .split(",")
        .map(x => x.trim())
        .filter(Boolean);

    loading.textContent = "Encoding ingredients…";
    const userEmb = await embed(query);

    loading.textContent = "Searching recipes…";

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

// =====================================
// INIT
// =====================================
async function init() {
    try {
        loading.textContent = "Loading tokenizer…";
        await loadTokenizer();

        loading.textContent = "Loading model…";
        await loadOnnxModel();

        loading.textContent = "Loading recipes…";
        await loadChunks();

        loading.textContent = "Ready ✔";
    } catch (e) {
        console.error(e);
        loading.textContent = "Error loading app";
    }
}

document.getElementById("searchBtn").onclick = recommend;
init();
