const TOTAL_CHUNKS = 17;

let recipes = [];
let loaded = false;

const loadDiv = document.getElementById("loading");
const progressDiv = document.getElementById("progress");

// Load JSON chunks
async function loadChunks() {
    if (loaded) return recipes;

    let all = [];
    for (let i = 1; i <= TOTAL_CHUNKS; i++) {
        progressDiv.textContent = `Loading chunk ${i}/${TOTAL_CHUNKS}…`;
        const chunk = await fetch(`chunks/part${i}.json`).then(r => r.json());
        all.push(...chunk);
    }

    progressDiv.textContent = "";
    loaded = true;
    recipes = all;

    console.log("Loaded recipes:", recipes.length);
    return recipes;
}

// TEXT → VECTOR (simple bag-of-words)
function textToVector(text) {
    const words = text.toLowerCase().split(/[\s,]+/);
    const vec = {};

    for (const w of words) {
        if (!w) continue;
        vec[w] = (vec[w] || 0) + 1;
    }

    return vec;
}

// Cosine similarity between bag-of-words and recipe embedding
function cosineBoW(userVec, recipeEmbedding) {
    // Convert user BoW to recipe embedding dimension
    const dim = recipeEmbedding.length;
    const arr = new Array(dim).fill(0);

    let idx = 0;
    for (const key in userVec) {
        // deterministic pseudo-hash to place word into vector
        const hashedIndex = Math.abs(hashString(key)) % dim;
        arr[hashedIndex] = userVec[key];
        idx++;
    }

    // Now cosine similarity
    let dot = 0, na = 0, nb = 0;

    for (let i = 0; i < dim; i++) {
        const a = arr[i];
        const b = recipeEmbedding[i];

        dot += a * b;
        na += a * a;
        nb += b * b;
    }

    if (na === 0 || nb === 0) return 0;
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = (hash * 31 + str.charCodeAt(i)) | 0;
    }
    return hash;
}

async function recommend() {
    const text = document.getElementById("ingredientsInput").value.trim();
    if (!text) return;

    loadDiv.textContent = "Loading recipes...";
    const list = await loadChunks();

    loadDiv.textContent = "Computing similarity…";

    // 1. encode user text
    const userVec = textToVector(text);

    // 2. compute score against MiniLM recipe embeddings
    list.forEach(r => {
        r.score = cosineBoW(userVec, r.embedding);
    });

    // 3. sort
    const top = list.sort((a,b) => b.score - a.score).slice(0,3);

    // render
    document.getElementById("results").innerHTML =
        top.map(r => `
        <div class="recipe-card">
            <h3>${r.cuisine.toUpperCase()}</h3>
            <b>Score:</b> ${r.score.toFixed(4)}<br>
            <b>Ingredients:</b> ${r.ingredients.join(", ")}
        </div>
    `).join("");

    loadDiv.textContent = "";
}

document.getElementById("searchBtn").onclick = recommend;
