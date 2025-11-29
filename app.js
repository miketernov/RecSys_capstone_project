// =======================
// IMPORT TRANSFORMERS.JS
// =======================
import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.1";

const loadingDiv = document.getElementById("loading");
const progressDiv = document.getElementById("progress");

let embedder;
let modelReady = false;

// Load MiniLM model
async function initModel() {
  loadingDiv.textContent = "Loading MiniLM model… (~10–20 sec)";

  embedder = await pipeline(
    "feature-extraction",
    "Xenova/paraphrase-MiniLM-L3-v2"
  );

  modelReady = true;
  loadingDiv.textContent = "Model loaded ✔";
}

initModel();

// cosine similarity
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// =============================
// LOAD JSON CHUNKS SEQUENTIALLY
// =============================
const TOTAL_CHUNKS = 17;   // <--- Поставь своё количество

async function loadChunks() {
  let allRecipes = [];

  for (let i = 1; i <= TOTAL_CHUNKS; i++) {
    progressDiv.textContent = `Loading chunk ${i}/${TOTAL_CHUNKS}…`;

    const url = `chunks/part${i}.json`;
    const chunk = await fetch(url).then(r => r.json());

    allRecipes = allRecipes.concat(chunk);
  }

  progressDiv.textContent = "";
  return allRecipes;
}

// =============================
// MAIN RECOMMEND FUNCTION
// =============================
async function recommend() {
  if (!modelReady) {
    alert("Model is still loading… wait 10–20 sec");
    return;
  }

  const text = document.getElementById("ingredientsInput").value.trim();
  if (!text) return;

  loadingDiv.textContent = "Embedding your ingredients…";

  const output = await embedder(text);
  const userEmbedding = Array.from(output.data[0]);

  loadingDiv.textContent = "Downloading recipe chunks…";

  const recipes = await loadChunks();

  loadingDiv.textContent = "Computing similarity…";

  recipes.forEach(r => {
    r.score = cosine(userEmbedding, r.embedding);
  });

  const top = recipes.sort((a, b) => b.score - a.score).slice(0, 3);

  document.getElementById("results").innerHTML = top.map(r => `
    <div class="recipe-card">
      <h3>${r.cuisine.toUpperCase()}</h3>
      <b>Score:</b> ${r.score.toFixed(4)}<br>
      <b>Ingredients:</b> ${r.ingredients.join(", ")}
    </div>
  `).join("");

  loadingDiv.textContent = "";
}

document.getElementById("searchBtn").onclick = recommend;
