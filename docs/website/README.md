# Gradgen Documentation Website

This directory contains a Docusaurus v3 documentation website for the Gradgen project.

## Installation

The website is already installed with all dependencies. If you need to reinstall:

```bash
npm install
```

## Development

Start the development server with hot reload:

```bash
npm start
```

The site will be available at `http://localhost:3000`.

## Building

Create a production build:

```bash
npm run build
```

The static files will be generated in the `build/` directory.

## Serving

To serve and test the production build locally:

```bash
npm run serve
```

## Structure

- `docs/` - Documentation pages
  - `intro.md` - Homepage/introduction
  - `guide/` - Getting started guide
  - `examples/` - Code examples
  - `api/` - API reference
- `blog/` - Blog posts
- `src/` - Custom React components and styling
- `static/` - Static assets (images, etc.)
- `docusaurus.config.ts` - Main configuration file
- `sidebars.ts` - Sidebar navigation configuration

## Configuration

Edit `docusaurus.config.ts` to customize:

- Site title and tagline
- Organization and repo links
- Navigation bar items
- Footer links
- Theme colors

## Adding Content

### New Documentation Pages

1. Create a new `.md` file in the appropriate `docs/` subdirectory
2. Add frontmatter with metadata:
   ```markdown
   ---
   sidebar_position: 1
   title: My Page Title
   ---
   ```
3. The page will automatically appear in the sidebar based on directory structure

### Blog Posts

1. Create a new folder in `blog/` with the date pattern: `YYYY-MM-DD-post-name`
2. Create `index.md` inside with frontmatter:
   ```markdown
   ---
   authors:
     - name: Author Name
   ---
   ```

## Deployment

To deploy to GitHub Pages or other hosting, follow the Docusaurus deployment guides at https://docusaurus.io/docs/deployment

## Customization

- Edit `src/css/custom.css` for custom styling
- Add React components in `src/components/`
- Modify the theme in `docusaurus.config.ts` under `themeConfig`

## Resources

- [Docusaurus Documentation](https://docusaurus.io/docs)
- [Markdown Features](https://docusaurus.io/docs/markdown-features)
- [API Documentation](https://docusaurus.io/docs/api/docusaurus)
