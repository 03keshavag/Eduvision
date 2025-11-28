export function setupButtons(onRandomElement, onToggleView) {
    const randomBtn = document.getElementById("randomBtn");
    const viewBtn = document.getElementById("viewBtn");

    randomBtn.addEventListener("click", onRandomElement);
    viewBtn.addEventListener("click", onToggleView);
}
