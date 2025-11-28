import { bounceWithinBounds } from "./utils.js";

export class AnimationEngine {

    constructor(electrons) {
        this.electrons = electrons;
        this.speed = 1;
        this.flowMode = false;
    }

    heat() {
        this.speed = Math.min(6, this.speed + 0.5);
    }

    cool() {
        this.speed = Math.max(0.5, this.speed - 0.5);
    }

    flow() {
        this.flowMode = !this.flowMode;
    }

    update(canvas) {
        this.electrons.forEach(e => {
            if (this.flowMode) {
                e.x += this.speed * 2;
            } else {
                e.x += e.speedX * this.speed;
                e.y += e.speedY * this.speed;
            }

            bounceWithinBounds(e, canvas);
        });
    }
}
