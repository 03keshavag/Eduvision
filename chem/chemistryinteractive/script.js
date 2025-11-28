const layout = [
"H", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "He",
"Li", "Be", "", "", "", "", "", "", "B", "C", "N", "O", "F", "Ne",
"Na", "Mg", "", "", "", "", "", "", "Al", "Si", "P", "S", "Cl", "Ar",
"K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
"Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
"Cs", "Ba", "", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
"Fr", "Ra", "", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
];

const container = document.getElementById("table");
const modal = document.getElementById("modal");
const closeBtn = document.getElementById("close");
const nameBox = document.getElementById("element-name");
const infoBox = document.getElementById("element-info");

fetch("./data/elements.json")
.then(res => res.json())
.then(data => {
    layout.forEach(symbol => {
        const box = document.createElement("div");

        if (symbol !== "") {
            box.className = "element";
            box.textContent = symbol;

            box.addEventListener("click", () => {
                nameBox.textContent = data[symbol].name;
                infoBox.textContent = data[symbol].info;
                modal.classList.remove("hidden");
            });
        }

        container.appendChild(box);
    });
});

closeBtn.addEventListener("click", () => modal.classList.add("hidden"));
modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.add("hidden");
});
