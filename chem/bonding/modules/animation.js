export class AnimationEngine {
    constructor() {
        this.progress = 0;
        this.active = false;
    }

    start() {
        this.progress = 0;
        this.active = true;
    }

    animate(atomA, atomB, electrons) {
        if (!this.active) return;

        this.progress += 0.02;
        if (this.progress >= 1) this.active = false;

        electrons.forEach(el => {
            el.x += (atomB.x - el.x) * 0.05;
            el.y += (atomB.y - el.y) * 0.05;
        });
    }
}
