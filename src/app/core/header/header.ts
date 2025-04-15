import {Header} from '@google/glue/lib/header';

module HeaderModule {
  function init(): void {
    const headerEl = document.querySelector<HTMLElement>('.glue-header');
    if (headerEl) new Header(headerEl);
  }

  init();
}
