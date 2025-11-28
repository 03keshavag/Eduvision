export class StateManager {
    constructor() {
        this.mode = "bohr"; // available: bohr | lewis | compare | edit
    }

    toggleMode() {
        if (this.mode === "bohr") this.mode = "lewis";
        else if (this.mode === "lewis") this.mode = "compare";
        else if (this.mode === "compare") this.mode = "edit";
        else this.mode = "bohr";
    }

    getMode() {
        return this.mode;
    }
}
