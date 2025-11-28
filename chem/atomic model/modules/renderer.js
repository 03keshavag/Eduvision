export class Renderer {
    constructor(ctx) {
        this.ctx = ctx;
    }

    drawNucleus() {
        this.ctx.fillStyle = "#ff3b3b";
        this.ctx.beginPath();
        this.ctx.arc(250, 250, 22, 0, Math.PI * 2);
        this.ctx.fill();
    }

    drawShell(radius) {
        this.ctx.strokeStyle = "#00ffaa";
        this.ctx.beginPath();
        this.ctx.arc(250, 250, radius, 0, Math.PI * 2);
        this.ctx.stroke();
    }

    drawElectron(x, y, excited) {
        this.ctx.fillStyle = excited ? "#ffaa00" : "#00ccff";
        this.ctx.beginPath();
        this.ctx.arc(x, y, 8, 0, Math.PI * 2);
        this.ctx.fill();
    }

    clear() {
        this.ctx.clearRect(0, 0, 500, 500);
    }
}
