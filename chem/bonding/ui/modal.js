export function setupModal() {
    const modal = document.getElementById("infoModal");
    const closeBtn = document.getElementById("closeModal");

    function show(text) {
        document.getElementById("modalContent").innerText = text;
        modal.classList.remove("hidden");
    }

    function hide() {
        modal.classList.add("hidden");
    }

    closeBtn.onclick = hide;
    modal.onclick = (e) => { if (e.target === modal) hide(); };

    return { show, hide };
}
