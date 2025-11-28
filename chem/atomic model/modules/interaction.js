export class InteractionHandler {
    constructor(canvas) {
        this.canvas = canvas;
        this.dragging = false;
    }

    enable() {
        this.canvas.addEventListener("mousedown", () => this.dragging = true);
        this.canvas.addEventListener("mouseup", () => this.dragging = false);
    }

    isDragging() {
        return this.dragging;
    }
}
