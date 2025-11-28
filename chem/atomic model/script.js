const canvas = document.getElementById("atomCanvas");
const ctx = canvas.getContext("2d");

let angle = 0;
let excited = false;

const electronConfig = {
    H: [1],
    He: [2],
    Li: [2, 1],
    C: [2, 4],
    O: [2, 6],
    Ne: [2, 8]
};

function drawAtom(element) {
    ctx.clearRect(0,0,500,500);

    const config = electronConfig[element];

    ctx.fillStyle = "#ff3b3b";
    ctx.beginPath();
    ctx.arc(250,250,20,0,Math.PI*2);
    ctx.fill();

    config.forEach((count,i)=>{
        const radius = 60 + i * 50;

        ctx.strokeStyle = "#00ffaa";
        ctx.beginPath();
        ctx.arc(250,250,radius,0,Math.PI*2);
        ctx.stroke();

        for(let j=0;j<count;j++){
            const step = (Math.PI * 2) / count;
            const x = 250 + radius * Math.cos(angle + j * step);
            const y = 250 + radius * Math.sin(angle + j * step);

            ctx.fillStyle = excited ? "#ffaa00" : "#00ccff";
            ctx.beginPath();
            ctx.arc(x,y,8,0,Math.PI*2);
            ctx.fill();
        }
    });

    angle += excited ? 0.05 : 0.02;
}

let selected = document.getElementById("elementSelect").value;

setInterval(()=> drawAtom(selected),30);

document.getElementById("elementSelect").addEventListener("change",(e)=>{
    selected = e.target.value;
    excited = false;
});

document.getElementById("exciteBtn").addEventListener("click",()=>{
    excited = true;
});

document.getElementById("resetBtn").addEventListener("click",()=>{
    excited = false;
});
