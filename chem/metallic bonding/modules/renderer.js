export class Renderer {

    constructor(ctx) {
        this.ctx = ctx;
    }

    clear() {
        this.ctx.clearRect(0, 0, 650, 450);
    }

    drawIon(ion) {
        this.ctx.fillStyle = "#00bfff";
        this.ctx.beginPath();
        this.ctx.arc(ion.x, ion.y, ion.radius, 0, Math.PI * 2);
        this.ctx.fill();

        this.ctx.fillStyle = "#001a33";
        this.ctx.font = "16px Arial";
        this.ctx.fillText("+", ion.x - 5, ion.y + 5);
    }

    drawElectron(electron) {
        this.ctx.shadowBlur = 10;
        this.ctx.shadowColor = "#ffaa00";
        this.ctx.fillStyle = "#ffd000";
        this.ctx.beginPath();
        this.ctx.arc(electron.x, electron.y, 6, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.shadowBlur = 0;
    }

    drawFlowArrow() {
        this.ctx.fillStyle = "#00ffaa";
        this.ctx.font = "20px Arial";
        this.ctx.fillText("→", 300, 420);
    }
}
