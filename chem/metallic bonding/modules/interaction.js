import { distance } from "./utils.js";

export class Interaction {

    constructor(canvas, electrons) {
        this.canvas = canvas;
        this.electrons = electrons;
        this.activeElectron = null;
        this._enableEvents();
    }

    _enableEvents() {

        this.canvas.addEventListener("mousedown", (e) => {
            const { offsetX, offsetY } = e;

            for (const electron of this.electrons) {
                if (distance(electron.x, electron.y, offsetX, offsetY) < 10) {
                    this.activeElectron = electron;
                    return;
                }
            }
        });

        this.canvas.addEventListener("mousemove", (e) => {
            if (this.activeElectron) {
                this.activeElectron.x = e.offsetX;
                this.activeElectron.y = e.offsetY;
            }
        });

        this.canvas.addEventListener("mouseup", () => {
            this.activeElectron = null;
        });
    }
}
