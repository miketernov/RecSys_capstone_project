// ======================================================
// CONFIG
// ======================================================

const MODEL_ID = "Xenova/all-MiniLM-L6-v2";   // используем ту же модель, что и в Colab
const TOTAL_CHUNKS = 17;                     // <-- поставь твоё количество partN.json

let embedder = null;
let recipes = [];
let modelReady = false;

const loadingDiv = document.getElementById("loading");
const progressDiv = document.getElementById("progress");


// ======================================================
// LOAD MODEL
// ======================================================

async function initModel() {
    loadingDiv.textContent = "Loading MiniLM model… (~10–20 sec)";

    embedder = await window.transformers.pipeline(
        "feature-extraction",
        MODEL_ID
    );

    modelReady = true;
    loadingDiv.textContent = "Model loaded ✔";
}

initModel();


// ======================================================
// SAFE COSINE SIMILARITY
// ======================================================

function cosine(a, b) {
    if (!a || !b || a.length !== b.length) return 0;

    let dot = 0, na = 0, nb = 0;

    for (let i = 0; i < a.length; i++) {
        const x = Number.isFinite(a[i]) ? a[i] : 0;
        const y = Number.isFinite(b[i]) ? b[i] : 0;

        dot += x * y;
        na += x * x;
        nb += y * y;
    }

    if (na === 0 || nb === 0) return 0;

    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}


// ======================================================
// LOAD CHUNKS + CLEAN NAN
// ======================================================

async function loadChunks() {
    if (recipes.length > 0) return recipes;

    let all = [];

    for (let i = 1; i <= TOTAL_CHUNKS; i++) {
        progressDiv.textContent = `Loading chunk ${i}/${TOTAL_CHUNKS}…`;

        const chunk = await fetch(`chunks/part${i}.json`).then(r => r.json());

        chunk.forEach(r => {
            // Чистим nan / null / undefined
            r.embedding = r.embedding.map(v =>
                Number.isFinite(v) ? v : 0
            );
        });

        all.push(...chunk);
    }

    progressDiv.textContent = "";

    recipes = all;
    console.log("Loaded recipes:", recipes.length);
    return recipes;
}


// ======================================================
// EMBED TEXT (WITH NAN CLEANING)
// ======================================================

async function embedText(text) {
    const out = await embedder(text, {
        pooling: "mean",
        normalize: true,
    });

    // out.data[0] = Float32Array
    let vec = Array.from(out.data[0]);

    // чистим
    vec = vec.map(v => Number.isFinite(v) ? v : 0);

    return vec;
}


// ======================================================
// MAIN RECOMMENDER
// ======================================================

async function recommend() {
    if (!modelReady) {
        alert("Model still loading… wait 5–10 seconds");
        return;
    }

    const text = document.getElementById("ingredientsInput").value.trim();
    if (!text) return;

    loadingDiv.textContent = "Embedding your ingredients…";
    const userEmbedding = await embedText(text);

    loadingDiv.textContent = "Loading recipes…";
    const data = await loadChunks();

    loadingDiv.textContent = "Computing similarity…";

    data.forEach(r => {
        r.score = cosine(userEmbedding, r.embedding);
    });

    const top = data
        .sort((a, b) => b.score - a.score)
        .slice(0, 3);

    document.getElementById("results").innerHTML =
        top.map(r => `
            <div class="recipe-card">
                <h3>${r.cuisine.toUpperCase()}</h3>
                <b>Score:</b> ${r.score.toFixed(4)}<br>
                <b>Ingredients:</b> ${r.ingredients.join(", ")}
            </div>
        `).join("");

    loadingDiv.textContent = "";
}


// ======================================================
// BUTTON
// ======================================================

document.getElementById("searchBtn").onclick = recommend;
