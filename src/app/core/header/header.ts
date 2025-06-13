import {Header} from '@google/glue/lib/header';

const enum Classes {
  isOpen = 'is-open',
}

const enum Selectors {
  dropdown = '[data-js-dropdown]',
  dropdownTrigger = '[data-js-dropdown-trigger]',
  dropdownTarget = '[data-js-dropdown-target]',
}

module HeaderModule {
  function init(): void {
    const headerEl = document.querySelector<HTMLElement>('.glue-header');
    if (headerEl) new Header(headerEl);

    const dropdownEls = headerEl?.querySelectorAll(Selectors.dropdown);

    dropdownEls?.forEach(el => bindDropdown(el as HTMLElement));
  }

  function bindDropdown(dropdown: HTMLElement) {
    const trigger = dropdown.querySelector(Selectors.dropdownTrigger);
    const target = dropdown.querySelector(Selectors.dropdownTarget);

    trigger?.addEventListener('click', e => {
      e.preventDefault();
      e.stopPropagation();

      target?.classList.toggle(Classes.isOpen);

      if (target?.classList.contains(Classes.isOpen)) {
        trigger.setAttribute('aria-expanded', 'true');
        document.body.addEventListener('click', onClickOutside);
      } else {
        trigger.setAttribute('aria-expanded', 'false');
        document.body.removeEventListener('click', onClickOutside);
      }
    });
  }

  function onClickOutside(e: MouseEvent) {
    if (!(e.target as HTMLElement)?.closest(Selectors.dropdown)) {
      const dropdowns = document.querySelectorAll(Selectors.dropdown);
      dropdowns.forEach(el => {
        el.querySelector(Selectors.dropdownTarget)?.classList.remove(Classes.isOpen);
        el.querySelector(Selectors.dropdownTrigger)?.setAttribute('aria-expanded', 'false');
        document.body.removeEventListener('click', onClickOutside);
      });
    }
  }

  init();
}
