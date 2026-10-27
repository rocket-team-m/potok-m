// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Сайт команды: https://rocket-team-m.github.io/potok-m/
export default defineConfig({
  site: 'https://rocket-team-m.github.io',
  base: '/potok-m',
  integrations: [
    starlight({
      title: 'Поток-М',
      description:
        'Инженерная документация команды «Поток-М»: конструкция, борт, ИИ, полёты и процессы.',
      defaultLocale: 'root',
      locales: {
        root: { label: 'Русский', lang: 'ru' },
      },
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/rocket-team-m/potok-m',
        },
      ],
      editLink: {
        baseUrl: 'https://github.com/rocket-team-m/potok-m/edit/main/docs/',
      },
      lastUpdated: true,
      sidebar: [
        {
          label: 'Проект',
          items: [
            { label: 'Цели и ТТХ', slug: 'project/goals' },
            { label: 'План сезона', slug: 'project/roadmap' },
            { label: 'Роли', slug: 'project/roles' },
          ],
        },
        {
          label: 'Борт',
          items: [
            { label: 'Структурная схема', slug: 'hardware/block-diagram' },
            { label: 'Компоненты (BOM)', slug: 'hardware/bom' },
            { label: 'Система спасения', slug: 'hardware/recovery' },
            { label: 'Система поиска', slug: 'hardware/search' },
          ],
        },
        {
          label: 'ИИ',
          items: [
            { label: 'Концепт', slug: 'ai/concept' },
            { label: 'Обучение', slug: 'ai/training' },
            { label: 'Телеметрия', slug: 'ai/telemetry' },
          ],
        },
        {
          label: 'Форма и аэродинамика',
          items: [
            { label: 'Чертежи и CAD', slug: 'vehicle/cad' },
            { label: 'Аэродинамика', slug: 'vehicle/aero' },
          ],
        },
        {
          label: 'Полёты',
          items: [
            { label: 'Протокол пуска', slug: 'flights/protocol' },
            { label: 'Журнал полётов', slug: 'flights/log' },
          ],
        },
        {
          label: 'Процессы',
          items: [
            { label: 'Как мы работаем', slug: 'process/work' },
            { label: 'Решения (ADR)', slug: 'process/decisions' },
            { label: 'Глоссарий', slug: 'process/glossary' },
          ],
        },
      ],
    }),
  ],
});
