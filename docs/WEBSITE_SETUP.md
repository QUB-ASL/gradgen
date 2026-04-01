# Gradgen Documentation Website Setup

A **Docusaurus v3** documentation website has been set up for Gradgen in `/docs/website/`.

## Quick Start

Navigate to the website directory:

```bash
cd docs/website
```

### Start Development Server

```bash
npm start
```

The site opens automatically at `http://localhost:3000/`

### Build for Production

```bash
npm run build
```

Output is in the `build/` directory.

### Preview Production Build

```bash
npm run serve
```

## What's Included

The website includes:

- **Welcome Page** (`intro.md`) - Overview of Gradgen
- **Getting Started Guide** (`guide/getting-started.md`) - Installation and basic concepts
- **Basic Examples** (`examples/basic-examples.md`) - Simple code samples
- **API Reference** (`api/index.md`) - Core classes and methods
- Default Docusaurus theme and styling
- Blog support (in `blog/` directory)

## File Structure

```
docs/website/
├── docs/              # Documentation pages
│   ├── intro.md       # Welcome/homepage
│   ├── guide/         # Guides
│   ├── examples/      # Code examples
│   ├── api/           # API reference
│   └── tutorial-*     # Default Docusaurus tutorials
├── blog/              # Blog posts
├── src/               # Custom components & styling
├── static/            # Images and static assets
├── docusaurus.config.ts  # Main configuration
└── sidebars.ts        # Sidebar navigation
```

## Adding Content

### New Documentation Page

1. Create a new `.md` file in an existing directory (e.g., `docs/guide/`)
2. Add frontmatter:
   ```markdown
   ---
   sidebar_position: 2
   ---
   # Page Title
   ```
3. Write your content in Markdown
4. The page auto-appears in the sidebar

### Blog Post

1. Create a folder in `blog/`: `YYYY-MM-DD-post-title`
2. Create `index.md` with:
   ```markdown
   ---
   authors:
     - name: Your Name
   ---
   # Post Title
   ```

## Configuration

Edit `docusaurus.config.ts` to customize:

- **Site title**: Change `title: 'Gradgen'`
- **Organization**: Update `organizationName` and `projectName`
- **GitHub link**: Modify navbar `href` to your repo
- **Colors/Theme**: Edit `themeConfig` section

## Deployment

For GitHub Pages:

1. Update `url` and `baseUrl` in `docusaurus.config.ts`
2. Run `npm run build`
3. Deploy the `build/` directory

Alternatively, Docusaurus supports deployment to Netlify, Vercel, and other platforms.

## Next Steps

1. **Expand documentation** with your content
2. **Add more examples** showing Gradgen features
3. **Create blog posts** about updates and features
4. **Deploy** to production when ready

## Resources

- [Docusaurus Documentation](https://docusaurus.io/docs)
- [Markdown Guide](https://docusaurus.io/docs/markdown-features)
- [Configuration Reference](https://docusaurus.io/docs/api/docusaurus-config)

---

**Ready to go!** Run `cd docs/website && npm start` to begin.
