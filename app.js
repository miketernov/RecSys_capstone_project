// ================= CONFIG =================
const TOTAL_CHUNKS = 17;
const USE_SINGLE_FILE = true;
const RECIPES_FILE = "recipes_all.json";
const ONNX_URL =
  "https://huggingface.co/iammik3e/recsys-minilm/resolve/main/model.onnx";

const DIET_EXCLUSIONS = {
  vegetarian: ["chicken", "beef", "pork", "fish", "meat", "bacon", "ham"],
  vegan: [
    "meat", "chicken", "beef", "fish",
    "milk", "cheese", "egg", "butter", "cream"
  ],
  nopork: ["pork", "bacon", "ham"]
};

let recipes = [];
let session = null;
let tokenizer = null;
let ort = null;

// UI
const loading = document.getElementById("loading");
const progress = document.getElementById("progress");
const results = document.getElementById("results");

// ================= ONNX RUNTIME =================
async function loadONNXRuntime() {
  try {
    const m = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.wasm.min.js");
    ort = m.default || m;
    if (ort?.InferenceSession) return ort;
  } catch {}

  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.wasm.min.js";
    s.onload = () => window.ort ? resolve(window.ort) : reject();
    document.head.appendChild(s);
  });
}

// ================= TOKENIZER =================
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
  return { ids: [cls, ...ids].slice(0,128), len: Math.min(ids.length+1,128) };
}

function makeTensor(ids) {
  const arr = new BigInt64Array(128);
  ids.forEach((v,i)=>arr[i]=BigInt(v));
  return new ort.Tensor("int64", arr, [1,128]);
}

// ================= MODEL =================
async function loadModel() {
  ort = await loadONNXRuntime();
  session = await ort.InferenceSession.create(ONNX_URL, { executionProviders:["wasm"] });
}

// ================= DATA =================
async function loadChunks() {
  if (USE_SINGLE_FILE) {
    try {
      recipes = await fetch(RECIPES_FILE).then(r=>r.json());
      return;
    } catch {}
  }
  const parts = await Promise.all(
    Array.from({length:TOTAL_CHUNKS},(_,i)=>
      fetch(`chunks/part${i+1}.json`).then(r=>r.json()).catch(()=>[])
    )
  );
  recipes = parts.flat();
}

// ================= EMBEDDING =================
async function embed(text) {
  const tok = tokenize("Ingredients: "+text);
  const mask = new BigInt64Array(128);
  for(let i=0;i<tok.len;i++) mask[i]=1n;

  const out = await session.run({
    input_ids: makeTensor(tok.ids),
    attention_mask: new ort.Tensor("int64",mask,[1,128]),
    token_type_ids: new ort.Tensor("int64",new BigInt64Array(128),[1,128])
  });

  const data = out[Object.keys(out)[0]].data;
  const hidden = out[Object.keys(out)[0]].dims[2];
  const emb = new Array(hidden).fill(0);

  for(let i=0;i<hidden;i++){
    for(let j=0;j<tok.len;j++) emb[i]+=data[j*hidden+i];
    emb[i]/=tok.len;
  }
  return emb;
}

// ================= UTILS =================
function cosine(a,b){
  let d=0,na=0,nb=0;
  for(let i=0;i<a.length;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i]}
  return d/(Math.sqrt(na)*Math.sqrt(nb));
}

function getRecipeImage(recipe){
  const q = recipe.ingredients.slice(0,3).join(" ");
  return `https://source.unsplash.com/featured/400x300/?food,${encodeURIComponent(q)}`;
}

// ================= SEARCH =================
async function recommend(){
  const query = ingredientsInput.value.trim();
  if(!query) return;

  let excluded=[];
  if(dietVegetarian.checked) excluded.push(...DIET_EXCLUSIONS.vegetarian);
  if(dietVegan.checked) excluded.push(...DIET_EXCLUSIONS.vegan);
  if(dietNoPork.checked) excluded.push(...DIET_EXCLUSIONS.nopork);

  loading.textContent="Encoding…";
  const userEmb = await embed(query);

  const filtered = recipes.filter(r =>
    !excluded.some(e => r.ingredients.join(" ").includes(e))
  );

  const scored = filtered.map(r=>({r,score:cosine(userEmb,r.embedding)}))
    .sort((a,b)=>b.score-a.score)
    .slice(0,parseInt(topN.value));

  results.innerHTML="";
  scored.forEach(x=>{
    results.innerHTML+=`
      <div class="recipe-card">
        <img src="${getRecipeImage(x.r)}">
        <div class="card-body">
          <h3>${x.r.cuisine.toUpperCase()}</h3>
          <div class="score">Similarity: ${x.score.toFixed(4)}</div>
          <div class="ingredients">${x.r.ingredients.join(", ")}</div>
        </div>
      </div>`;
  });
  loading.textContent="";
}

// ================= INIT =================
async function init(){
  loading.textContent="Loading…";
  await loadTokenizer();
  await loadModel();
  await loadChunks();
  loading.textContent="Ready ✓";
}

searchBtn.onclick = recommend;
init();
