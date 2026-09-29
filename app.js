const filters = document.querySelectorAll('.filter');
const papers = document.querySelectorAll('.paper');
filters.forEach(button => button.addEventListener('click', () => {
  filters.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  papers.forEach(paper => {
    const show = button.dataset.filter === 'all' || paper.dataset.categories.split(' ').includes(button.dataset.filter);
    paper.hidden = !show;
  });
}));
const dialog = document.querySelector('.teaser-dialog');
const dialogImage = dialog.querySelector('img');
const dialogTitle = dialog.querySelector('h2');
const dialogSource = dialog.querySelector('.dialog-source');
let opener;
document.querySelectorAll('.teaser-button').forEach(button => button.addEventListener('click', () => {
  opener = button;
  dialogImage.src = button.querySelector('img').src;
  dialogImage.alt = button.querySelector('img').alt;
  dialogTitle.textContent = button.dataset.title;
  dialogSource.href = button.dataset.paper;
  dialog.querySelector('.figure-number').textContent = button.dataset.figure;
  dialog.showModal();
}));
dialog.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close(); } });
dialog.addEventListener('close', () => opener?.focus());
