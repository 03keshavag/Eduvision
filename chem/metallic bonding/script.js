import { Renderer } from "./modules/renderer.js";
import { AnimationEngine } from "./modules/animation.js";
import { Interaction } from "./modules/interaction.js";
import { setupButtons } from "./ui/buttons.js";
import { setupModal } from "./ui/modal.js";
import { startQuiz } from "./ui/quiz.js";

const canvas = document.getElementById("metalCanvas");
const ctx = canvas.getContext("2d");

let electrons = [];
let ions = [];
let selected = "Na";

const modal = setupModal();

const renderer = new Renderer(ctx);
const animation = new AnimationEngine(electrons);
let interaction = new Interaction(canvas, electrons);

async function loadMetal() {
    const res = await fetch("./data/metals.json");
    const metals = await res.json();
    const metal = metals[selected];

    ions = [];
    electrons = [];

    let spacing = 100;

    for (let i = 0; i < 4; i++){
        for (let j = 0; j < 6; j++){
            ions.push({ x: spacing + j * 80, y: spacing + i * 60, radius: 20 });
        }
    }

    for (let i = 0; i < metal.electrons; i++){
        electrons.push({
            x: Math.random()*canvas.width,
            y: Math.random()*canvas.height,
            speedX: Math.random()*2 -1,
            speedY: Math.random()*2 -1
        });
    }

    interaction = new Interaction(canvas, electrons);
}

document.getElementById("metalSelect").onchange = (e)=> {
    selected = e.target.value;
    loadMetal();
};

setupButtons(
    ()=> animation.heat(),
    ()=> animation.cool(),
    ()=> animation.flow()
);

document.getElementById("quizBtn").addEventListener("click", ()=> {
    startQuiz(modal.show);
});

loadMetal();

function loop(){
    renderer.clear();
    ions.forEach(i=> renderer.drawIon(i));
    electrons.forEach(e=> renderer.drawElectron(e));
    animation.update(canvas);
    requestAnimationFrame(loop);
}
loop();
