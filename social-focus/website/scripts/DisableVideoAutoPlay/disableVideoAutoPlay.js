(function () {
  let autoPlayVideoPlayer = null;
  let autoPlayChannelVideoPlayer = null;

  const videoAutoPlayRenderConfig = {
    childList: true,
    subtree: true,
  };

  browser.runtime.onMessage.addListener((message) => {
    if (
      message.id === "socialFocus_youtube_hide_thumbnails" &&
      message.type === "toggle"
    ) {
      const isDisableAutoPlay = document.documentElement.getAttribute(
        "socialFocus_youtube_hide_thumbnails"
      );

      if (isDisableAutoPlay) {
        // Pause current Video which user stop scrolling and on autoplay disable options
        if (autoPlayVideoPlayer) {
          autoPlayVideoPlayer.pause();
        }

        if (autoPlayChannelVideoPlayer) {
          autoPlayChannelVideoPlayer.pause();
        }

        // Start observer else videos logic
        videoAutoPlayRenderObserver.observe(
          document.body,
          videoAutoPlayRenderConfig
        );
      } else {
        // Play current Video which user stop scrolling and off autoplay disable options
        if (autoPlayVideoPlayer) {
          autoPlayVideoPlayer.play();
        }

        // Stop observer else videos logic
        videoAutoPlayRenderObserver.disconnect();
        channelVideoAutoPlayRenderObserver.disconnect();
      }
    }
  });

  const videoAutoPlayRenderObserver = new MutationObserver((mutations) => {
    mutations.forEach(() => {
      const youtubeAutoPlayVideoPlayer = document.querySelector(
        `${
          isUserAgentMobile() ? "ytm-video-preview" : "ytd-video-preview"
        } video`
      );

      if (youtubeAutoPlayVideoPlayer) {
        autoPlayVideoPlayer = youtubeAutoPlayVideoPlayer;

        const isDisableAutoPlay = document.documentElement.getAttribute(
          "socialFocus_youtube_hide_thumbnails"
        );

        if (isDisableAutoPlay === "true") {
          youtubeAutoPlayVideoPlayer.pause();
          youtubeAutoPlayVideoPlayer.muted = true;
        } else {
          youtubeAutoPlayVideoPlayer.play();
          youtubeAutoPlayVideoPlayer.muted = false;
          videoAutoPlayRenderObserver.disconnect();
        }
      }
    });
  });

  const channelVideoAutoPlayRenderObserver = new MutationObserver(
    (mutations) => {
      mutations.forEach(() => {
        const channelVideoVideoPlayer = document.querySelector(
          "ytd-channel-video-player-renderer video"
        );

        if (channelVideoVideoPlayer && !isUserAgentMobile()) {
          autoPlayChannelVideoPlayer = channelVideoVideoPlayer;

          const isDisableAutoPlay = document.documentElement.getAttribute(
            "socialFocus_youtube_hide_thumbnails"
          );

          if (isDisableAutoPlay === "true") {
            channelVideoVideoPlayer.pause();
            channelVideoVideoPlayer.muted = true;

            const timeout = setTimeout(() => {
              channelVideoAutoPlayRenderObserver.disconnect();
            }, 2000);

            clearTimeout(timeout);
          } else {
            channelVideoVideoPlayer.play();
            channelVideoVideoPlayer.muted = false;
            channelVideoAutoPlayRenderObserver.disconnect();
          }
        }
      });
    }
  );

  document.addEventListener("DOMContentLoaded", () => {
    const autoPlayEnabledAttribute = document.documentElement.getAttribute(
      "socialFocus_youtube_hide_thumbnails"
    );

    if (autoPlayEnabledAttribute === "true") {
      videoAutoPlayRenderObserver.observe(
        document.body,
        videoAutoPlayRenderConfig
      );

      channelVideoAutoPlayRenderObserver.observe(
        document.body,
        videoAutoPlayRenderConfig
      );
    } else {
      videoAutoPlayRenderObserver.disconnect();
      channelVideoAutoPlayRenderObserver.disconnect();
    }
  });
})();
