---
layout: post
title: "Game design: one finger to rule them all"
tags: gamedev devlog scape gamedesign
---
Nowadays, smartphones are everywhere. People are swiping up and down, left and right, panning in, double tapping like champions. What people don't want is having to use more than a single finger, god forbit a second hand just to get a task done on their device. Game developers such as [King](https://king.com) have taken that principle to another level for a simple reason: **Accessibility**.

If you want your game to be played by the masses, it needs to be accessible. In order for a game to be accessible, it needs to be supported by the hardware and the user should not get confused how to actually play the game. Personally, I always disliked the fact that controls are being emulated on smartphones via HUDs like this:

![fortnite-hud](https://icdn9.digitaltrends.com/image/fortnite-mobile-beginners-guide-gather-720x720.jpg)

Especially on a busy train or a cigarette in one hand, playing those kind of games can be quite tricky. Why don't all mobile games just have simple **One Finger to Rule them All** controls?

# Simple controls are challenging

As a user I do not want to read through manuals or tutorials to learn how to actually play the game. Time is much better spent and there are so many games out there which do not require any tutorials whatsoever. Thus, designing a simple input system which can be just with just one finger is rather challenging:

* how does the player know if he should swipe, pan or where to click?
* how do I prevent that the player accidentally uses wrong controls?
* how can I ensure the player learns the controls naturally by just trying out?

> the more limited the controls are, the more accessible the game is. However, the amount of input combinations decreases with limited controls.

Finding the perfect balance between those two is the real challenge.

# A first approach

Currently I am working on a small game called **scape** - it is a fast-paced 2D platformer written in [Java](https://en.wikipedia.org/wiki/Java_(programming_language)), using my gamejam framework called [braingdx](https://github.com/bitbrain/braingdx). You play a little virus infecting a compuer system. I got inspired by [Yoo Ninja!](https://yoo-ninja-free.en.uptodown.com/android), one of my favourite Android games:

![yoo-ninja](https://img.utdstc.com/screen/13/yoo-ninja-free-1.jpg:l)

Basically, the idea is to reach the end of the level without falling out of bounds. Touch the screen to jump (and effectively flip gravity). This is how my game **scape** loks like:

<video controls autoplay preload="auto" playsinline="" poster="https://pbs.twimg.com/tweet_video_thumb/D3dLC8CW4AAHcjP.jpg" src="https://video.twimg.com/tweet_video/D3dLC8CW4AAHcjP.mp4" type="video/mp4" style="width: 100%; height: 100%; position: absolute; background-color: black; top: 0%; left: 0%; transform: rotate(0deg) scale(1.005);"></video>

The first thing the player does is touching the screen and one notices that the character will jump as a consequence. However, this has some impact on the initial game design:

* the player should not be punished for not touching the screen initially
* the player should notice that he needs to do _something_ in order to progress
* the player should also learn in the beginning what the consequences are if no action is taken

To solve all these questions I did a simple trick: I placed a block in front of the player. As a result the player bumps into the block at some point, gets stuck and the moving camera will kill the player if out of bounds.

