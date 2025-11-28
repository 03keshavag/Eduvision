export function setupToolBar(onElementSelect) {
    const elementA = document.getElementById("elementA");
    const elementB = document.getElementById("elementB");

    elementA.onchange = () => onElementSelect(elementA.value, elementB.value);
    elementB.onchange = () => onElementSelect(elementA.value, elementB.value);
}
