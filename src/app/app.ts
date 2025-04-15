// UI Components

import '../styles/styles.scss';

window.bundleError = false;

window.handleBundleError = () => {
  if (!window.bundleError) {
    window.bundleError = true;
    const notice = document.querySelector('[data-js-notice]');
    notice?.classList.remove('is-hidden');
  }
};

function loadScript() {
  const script = document.createElement('script');
  script.type = 'text/javascript';
  script.src = './core_init.bundle.js';
  script.onerror = () => window.handleBundleError();
  document.head.appendChild(script);
}

loadScript();
