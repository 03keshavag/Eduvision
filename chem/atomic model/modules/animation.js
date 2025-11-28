export class AnimationEngine {
    constructor() {
        this.angle = 0;
        this.speed = 0.02;
        this.excited = false;
    }

    update() {
        this.angle += this.excited ? 0.05 : this.speed;
    }

    excite() {
        this.excited = true;
    }

    reset() {
        this.excited = false;
    }
}
