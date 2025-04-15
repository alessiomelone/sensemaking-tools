# Sensemaker

Sensemaker Microsite

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project to a remote environment.

### Dependencies

- Node: `22.x.x`

### Setup

- Run `npm install` to install the required dependancies.

## Development

- Run `npm run dev` to spin up Webpack and watch for changes.
- Run `npm run preview` to spin up a local web server.
- Navigate to `http://localhost:8080` once the initial build is complete.

## Deployment

### Pages setup
- Settings -> Environments -> github-pages
- Deployment branches and tags -> Add deployment branch or tag rule
- Add "site" branch
- Settings -> Pages
- Build and deployment Source -> "GitHub Actions"

### Overview
- This app is built/deployed using GitHub Actions
- As part of this process the following occurs:
  - The app in _this_ branch is built (known as "Site")
  - The "Docs" app from the `/docs` directory on the `main` branch is copied into a subdirectory of "Site"
  - Both sites are deployed and served via GitHub Pages

### Deploying "Site" changes
- Any new commits "pushed" to the `site` branch will automatically trigger a new build/deployment.
- The current version of "Docs" will also be deployed at that same time.

### Deploying "Docs" changes
- In order to deploy changes to the "Docs" web app, you will need to commit and push a change to the `site` branch.
- We recommend you do this by editing the `/trigger-docs-deploy.txt` file with the current timestamp.
- Once this commit is pushed, the latest versions of both the "Site" and "Docs" will be deployed.
