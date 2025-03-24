const video = document.getElementById("video");
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");
document.body.appendChild(canvas);
canvas.style.position = "absolute";
canvas.style.top = video.offsetTop + "px";
canvas.style.left = video.offsetLeft + "px";

let keypointBuffer = [];
const SEQUENCE_LENGTH = 30;
let sendInterval = 700;

let lastStaticPrediction = "";
let stableStaticPrediction = "";
let staticConsistentCount = 0;
const REQUIRED_STATIC_STABILITY = 3;

const maxHistory = 5;
const historyList = [];
let collectedWord = "";
let wordBuffer = "";

function updateHistory(pred) {
  if (historyList.length >= maxHistory) {
    historyList.shift();
  }
  historyList.push(pred);
  const historyContainer = document.getElementById("history");
  historyContainer.innerHTML = "";
  for (let item of historyList.slice().reverse()) {
    const li = document.createElement("li");
    li.textContent = item;
    historyContainer.appendChild(li);
  }
}

function updateCollectedWord() {
  document.getElementById("collectedWord").innerText = collectedWord;
}

function deleteLastChar() {
  collectedWord = collectedWord.slice(0, -1);
  updateCollectedWord();
}
