// =====================================
// CONFIG
// =====================================
const TOTAL_CHUNKS = 17;
let recipes = [];
let vocabulary = new Map();   // word → index
let idf = [];                 // idf vector
let dim = 0;
let tfidfRecipes = [];        // TF-IDF vectors for recipes

const loading = document.getElementById("loading");
const progress = document.getElementById("progress");

// =====================================
// 1) LOAD RECIPES
// =====================================
async function loadChunks() {
    if (recipes.length > 0) return recipes;

    let all = [];
    for (let i = 1; i <= TOTAL_CHUNKS; i++) {
        progress.textContent = `Loading chunk ${i}/${TOTAL_CHUNKS}…`;
        const chunk = await fetch(`chunks/part${i}.json`).then(r => r.json());
        all.push(...chunk);
    }

    recipes = all;
    progress.textContent = "";
    return recipes;
}

// =====================================
// 2) BUILD VOCABULARY
// =====================================
function buildVocabulary(data) {
    const vocabSet = new Set();

    data.forEach(r => {
        r.ingredients.forEach(ing => {
            const tokens = ing.toLowerCase().split(/[\s,()\/-]+/);
            for (let t of tokens) {
                if (t.length > 1) vocabSet.add(t);
            }
        });
    });

    let index = 0;
    vocabSet.forEach(word => vocabulary.set(word, index++));
    dim = vocabulary.size;

    console.log("Vocabulary size =", dim);
}

// =====================================
// 3) COMPUTE IDF VALUES
// =====================================
function computeIDF(data) {
    const docCount = data.length;
    idf = new Array(dim).fill(0);

    data.forEach(r => {
        const docWords = new Set();
        r.ingredients.forEach(ing => {
            const tokens = ing.toLowerCase().split(/[\s,()\/-]+/);
            tokens.forEach(t => {
                if (vocabulary.has(t)) docWords.add(t);
            });
        });

        docWords.forEach(w => {
            const idx = vocabulary.get(w);
            idf[idx] += 1;
        });
    });

    idf = idf.map(df => Math.log((docCount + 1) / (df + 1)) + 1);
}

// =====================================
// 4) TF-IDF FOR ONE DOCUMENT
// =====================================
function tfidfVector(ingredients) {
    const vec = new Array(dim).fill(0);

    const freq = new Map();
    ingredients.forEach(ing => {
        const tokens = ing.toLowerCase().split(/[\s,()\/-]+/);
        for (let t of tokens) {
            if (vocabulary.has(t)) {
                freq.set(t, (freq.get(t) || 0) + 1);
            }
        }
    });

    freq.forEach((count, word) => {
        const idx = vocabulary.get(word);
        vec[idx] = count * idf[idx];
    });

    return vec;
}

// =====================================
// 5) BUILD TF-IDF FOR ALL RECIPES
// =====================================
function buildTFIDF(data) {
    tfidfRecipes = data.map(r => tfidfVector(r.ingredients));
    console.log("Built TF-IDF recipe vectors.");
}

// =====================================
// Cosine similarity
// =====================================
function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na === 0 || nb === 0) return 0;
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// =====================================
// MAIN SEARCH
// =====================================
async function recommend() {
    const txt = document.getElementById("ingredientsInput").value.trim();
    if (!txt) return;

    loading.textContent = "Building TF-IDF…";

    const userIngredients = txt.split(/[\s,]+/);
    const userVec = tfidfVector(userIngredients);

    const scores = recipes.map((r, idx) => ({
        recipe: r,
        score: cosine(userVec, tfidfRecipes[idx])
    }));

    const top = scores.sort((a, b) => b.score - a.score).slice(0, 3);

    document.getElementById("results").innerHTML = top.map(x => `
        <div class="recipe-card">
            <h3>${x.recipe.cuisine.toUpperCase()}</h3>
            <b>Score:</b> ${x.score.toFixed(4)}<br>
            <b>Ingredients:</b> ${x.recipe.ingredients.join(", ")}
        </div>
    `).join("");

    loading.textContent = "";
}

// =====================================
// INIT PIPELINE
// =====================================
async function init() {
    const data = await loadChunks();

    loading.textContent = "Building vocabulary…";
    buildVocabulary(data);

    loading.textContent = "Computing IDF…";
    computeIDF(data);

    loading.textContent = "Building TF-IDF vectors…";
    buildTFIDF(data);

    loading.textContent = "Ready ✔";
}
init();

document.getElementById("searchBtn").onclick = recommend;
