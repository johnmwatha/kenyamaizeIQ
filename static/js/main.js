// ── MaizeIQ Kenya — Main JS ──

// Add this to your existing script
document.addEventListener('DOMContentLoaded', function() {
    const wholesaleGroup = document.getElementById('wholesale_group');
    const retailGroup = document.getElementById('retail_group');
    const wholesaleRadio = document.getElementById('price_wholesale');
    const retailRadio = document.getElementById('price_retail');
    
    function togglePriceInputs() {
        if (wholesaleRadio.checked) {
            wholesaleGroup.style.display = 'block';
            retailGroup.style.display = 'none';
        } else {
            wholesaleGroup.style.display = 'none';
            retailGroup.style.display = 'block';
        }
    }
    
    wholesaleRadio.addEventListener('change', togglePriceInputs);
    retailRadio.addEventListener('change', togglePriceInputs);
    togglePriceInputs();
});
// Navbar scroll effect
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar?.classList.toggle('scrolled', window.scrollY > 20);
});

// Mobile menu toggle
function toggleMenu() {
  const links   = document.getElementById('navLinks');
  const actions = document.querySelector('.nav-actions');
  links?.classList.toggle('open');
  actions?.classList.toggle('open');
}

// Close menu on outside click
document.addEventListener('click', (e) => {
  if (!e.target.closest('.nav-inner')) {
    document.getElementById('navLinks')?.classList.remove('open');
    document.querySelector('.nav-actions')?.classList.remove('open');
  }
});

// Animate numbers on homepage stats
function animateCounter(el, target, duration = 1500) {
  const start = performance.now();
  const update = (time) => {
    const elapsed = time - start;
    const progress = Math.min(elapsed / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.floor(ease * target).toLocaleString();
    if (progress < 1) requestAnimationFrame(update);
    else el.textContent = target.toLocaleString();
  };
  requestAnimationFrame(update);
}

document.querySelectorAll('[data-count]').forEach(el => {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animateCounter(el, parseInt(el.dataset.count));
        observer.unobserve(el);
      }
    });
  }, { threshold: 0.5 });
  observer.observe(el);
});

// Fade-in on scroll
const fadeEls = document.querySelectorAll('.fade-in');
const fadeObserver = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.style.animationDelay = e.target.dataset.delay || '0s';
      e.target.classList.add('visible');
      fadeObserver.unobserve(e.target);
    }
  });
}, { threshold: 0.1 });
fadeEls.forEach(el => fadeObserver.observe(el));

// User dropdown toggle
function toggleUserMenu(e) {
  e.stopPropagation();
  document.getElementById('userDropdown')?.classList.toggle('open');
}
document.addEventListener('click', () => {
  document.getElementById('userDropdown')?.classList.remove('open');
});

// Tab system
function switchTab(tabId, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(tabId)?.classList.add('active');
  btn.classList.add('active');
}
