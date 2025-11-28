export class Renderer {
    constructor(ctx) {
        this.ctx = ctx;
    }

    clear() {
        this.ctx.clearRect(0, 0, 600, 450);
    }

    drawAtom(atom, highlight = false) {
        this.ctx.fillStyle = highlight ? "#00ff9d" : "#00bfff";
        this.ctx.beginPath();
        this.ctx.arc(atom.x, atom.y, atom.radius, 0, Math.PI * 2);
        this.ctx.fill();

        this.ctx.fillStyle = "black";
        this.ctx.font = "20px Arial";
        this.ctx.fillText(atom.symbol, atom.x - 10, atom.y + 7);
    }

    drawElectron(electron) {
        this.ctx.fillStyle = electron.shared ? "#ffdd00" : "#ffffff";
        this.ctx.beginPath();
        this.ctx.arc(electron.x, electron.y, 8, 0, Math.PI * 2);
        this.ctx.fill();
    }

    drawBondLine(atomA, atomB) {
        this.ctx.strokeStyle = "#ffaa00";
        this.ctx.lineWidth = 4;
        this.ctx.beginPath();
        this.ctx.moveTo(atomA.x, atomA.y);
        this.ctx.lineTo(atomB.x, atomB.y);
        this.ctx.stroke();
    }
}
