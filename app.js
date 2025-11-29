let embedder;
let modelReady = false;

const loadingDiv = document.getElementById("loading");

async function loadModel() {
  loadingDiv.innerText = "Loading model… (10–20 seconds)";

  // UMD API:
  embedder = await window.Transformers.pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );

  modelReady = true;
  loadingDiv.innerText = "Model loaded ✔";
}

loadModel();

// cosine
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

async function recommend() {
  if (!modelReady) {
    alert("Model still loading...");
    return;
  }

  const text = document.getElementById("ingredientsInput").value.trim();
  if (!text) return;

  loadingDiv.innerText = "Embedding user input…";

  const output = await embedder(text);
  const userEmbedding = Array.from(output.data[0]);

  loadingDiv.innerText = "Downloading recipes…";

  const recipesURL =
    "https://github.com/miketernov/RecSys_capstone_project/releases/download/v1/recipes_with_embeddings.json";

  const recipes = await fetch(recipesURL).then(r => r.json());

  loadingDiv.innerText = "Computing similarity…";

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

  loadingDiv.innerText = "";
}

document.getElementById("searchBtn").onclick = recommend;
