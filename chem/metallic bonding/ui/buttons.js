export function setupButtons(onHeat,onCool,onFlow){
    document.getElementById("heatBtn").onclick = onHeat;
    document.getElementById("coolBtn").onclick = onCool;
    document.getElementById("flowBtn").onclick = onFlow;
}
