export function setupButtons(onReset, onAutoBond, onNextChallenge) {
    document.getElementById("reset").addEventListener("click", onReset);
    document.getElementById("autoBond").addEventListener("click", onAutoBond);
    document.getElementById("nextChallenge").addEventListener("click", onNextChallenge);
}
