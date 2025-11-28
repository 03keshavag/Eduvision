import { distance } from "./utils.js";

export class BondingEngine {
    constructor() {}

    checkIonic(atomA, atomB, electrons) {
        const total = electrons.length;
        return Math.abs(atomA.valence - atomB.valence) === 1 && total <= atomA.valence + atomB.valence;
    }

    checkCovalent(atomA, atomB, electrons) {
        // simple rule: both need electrons and share near midpoint
        return electrons.some(e => 
            distance(e.x, e.y, (atomA.x + atomB.x)/2, (atomA.y + atomB.y)/2) < 40
        );
    }

    determineBond(atomA, atomB, electrons) {
        if (this.checkIonic(atomA, atomB, electrons)) return "ionic";
        if (this.checkCovalent(atomA, atomB, electrons)) return "covalent";
        return "none";
    }
}
