// Keeps scratch files out of the packaged extension.
module.exports = {
  ignoreFiles: [
    'tmp',
    'tmp/**/*',
    'web-ext-config.cjs',
    '**/.DS_Store',
  ],
};
