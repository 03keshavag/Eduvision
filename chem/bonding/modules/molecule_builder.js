import { midpoint } from "./utils.js";

export class MoleculeBuilder {
    constructor() {}

    alignCovalent(atomA, atomB, electrons) {
        const mid = midpoint(atomA.x, atomA.y, atomB.x, atomB.y);

        electrons.forEach(el => {
            el.x = mid.x + Math.random() * 20 - 10;
            el.y = mid.y + Math.random() * 20 - 10;
            el.shared = true;
        });
    }

    alignIonic(atomA, atomB, electrons) {
        electrons.forEach(el => {
            el.x = atomB.x + Math.random() * 40 - 20;
            el.y = atomB.y + Math.random() * 40 - 20;
        });
    }
}
