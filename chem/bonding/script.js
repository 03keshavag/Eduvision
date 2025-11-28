const canvas = document.getElementById("bondCanvas");
const ctx = canvas.getContext("2d");

// basic element valence electrons (simple for demo)
const valenceData = {
    H: 1, O: 6, C: 4, Na: 1, Cl: 7
};

let electrons = [];
let elements = [];

function spawnElement(symbol, x, y) {
    return {
        symbol,
        x,
        y,
        radius: 40,
        valence: valenceData[symbol]
    };
}

function spawnElectronsFor(atom) {
    electrons = [];
    const radius = 70;
    for (let i = 0; i < atom.valence; i++) {
        const angle = (Math.PI * 2 * i) / atom.valence;
        electrons.push({
            x: atom.x + radius * Math.cos(angle),
            y: atom.y + radius * Math.sin(angle),
            dragging: false,
            owner: atom.symbol
        });
    }
}

function drawAtom(atom) {
    ctx.font = "22px Arial";
    ctx.fillStyle = "#00eaff";
    ctx.beginPath();
    ctx.arc(atom.x, atom.y, atom.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#000";
    ctx.fillText(atom.symbol, atom.x - 12, atom.y + 8);
}

function drawElectron(e) {
    ctx.fillStyle = "#ffd900";
    ctx.beginPath();
    ctx.arc(e.x, e.y, 10, 0, Math.PI * 2);
    ctx.fill();
}

function reset() {
    electrons = [];
    elements = [];
    updateSelection();
}

function updateSelection() {
    const elA = document.getElementById("elementA").value;
    const elB = document.getElementById("elementB").value;

    elements = [
        spawnElement(elA, 200, 225),
        spawnElement(elB, 400, 225)
    ];
    spawnElectronsFor(elements[0]);
    spawnElectronsFor(elements[1]);
}

function autoBond() {
    // simple snap midpoint bond representation
    electrons.forEach(e => {
        e.x = 300;
        e.y = 225;
    });
}

canvas.addEventListener("mousedown", (event) => {
    const { offsetX, offsetY } = event;

    electrons.forEach(e => {
        const dist = Math.hypot(e.x - offsetX, e.y - offsetY);
        if (dist < 12) e.dragging = true;
    });
});

canvas.addEventListener("mousemove", (event) => {
    const { offsetX, offsetY } = event;
    electrons.forEach(e => {
        if (e.dragging) {
            e.x = offsetX;
            e.y = offsetY;
        }
    });
});

canvas.addEventListener("mouseup", () => {
    electrons.forEach(e => e.dragging = false);
});

document.getElementById("reset").onclick = reset;
document.getElementById("autoBond").onclick = autoBond;
document.getElementById("elementA").onchange = updateSelection;
document.getElementById("elementB").onchange = updateSelection;

updateSelection();

function loop() {
    ctx.clearRect(0,0,canvas.width,canvas.height);

    elements.forEach(drawAtom);
    electrons.forEach(drawElectron);

    requestAnimationFrame(loop);
}

loop();
