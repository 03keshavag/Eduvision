import { distance } from "./utils.js";

export class Interaction {
    constructor(canvas, electrons) {
        this.canvas = canvas;
        this.electrons = electrons;
        this.dragging = null;

        this._enable();
    }

    _enable() {
        this.canvas.addEventListener("mousedown", (e) => {
            const { offsetX, offsetY } = e;

            for (const el of this.electrons) {
                if (distance(el.x, el.y, offsetX, offsetY) < 12) {
                    this.dragging = el;
                }
            }
        });

        this.canvas.addEventListener("mousemove", (e) => {
            if (this.dragging) {
                this.dragging.x = e.offsetX;
                this.dragging.y = e.offsetY;
            }
        });

        this.canvas.addEventListener("mouseup", () => {
            this.dragging = null;
        });
    }
}
