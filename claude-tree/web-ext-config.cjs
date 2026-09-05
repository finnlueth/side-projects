// Keeps everything that is not the extension out of the packaged build.
module.exports = {
  ignoreFiles: [
    'test',
    'test/**/*',
    'tmp',
    'tmp/**/*',
    'web-ext-config.cjs',
    '_check.html',
    '**/.DS_Store',
  ],
};
