---
layout: post
title: "Building a game from scratch in 48 hours"
description: "I built a game in 48 hours from scratch. Let me show you how I did this."
tags: minibeansjam libgdx gamedev pixelart
---

[![minibeansjam5](/public/media/minibeansjam5.gif)](https://itch.io/jam/minibeansjam5)

Between Friday 3rd-5th April 2020, German [Rocketbeans TV](https://rocketbeans.tv/) hosted [minibeansjam 5](https://itch.io/jam/minibeansjam5), a 48 hour gamejam. I submitted a game and managed to finish most bits right in time. 

The purpose is not to win anything, it's more a fun thing to do and it really challenges your skills in time management, setting priorities as well as health care. (yes, you **NEED** sleep!)

# Preparation

Personally, I really need to be in the right mood to approach something like a gamejam. I am a quite emotional developer and if I feel down or unwell, my creativity kinda stalls. That's why I actually do not really _prepare_ for a gamejam but I really need to be ready for it. As a general rule though, please consider the following before approaching **any** gamejam:

* **gamejams are there to have fun**. Do not stress yourself! I know it can quite daunting at first, however we're all in the same boat. Enjoy it!
* **fix your sleep cycle!** There is nothing worse than a broken sleep cycle. I know, people have their preferred time to stay awake (some people even like to work only during night). This is completely fine. However, make sure you keep it consistent, since otherwise you might oversleep on the last day and miss the deadline.
* **stay hydrated!** prepare yourself with your favourite drink (sparkling water or still water is my favourite pick) - this keeps the braincells active and moist!
* **Stay Away From Energy Drinks (SAFED)**
* **sharpen your tools** on kick-off day, make sure you have everything already opened, preloaded etc. so you do not have to do it once the themes are announced.
* **Together is Better!** find people who want to do it with you. I promise you it changes the entire experience.

Interestingly, I purposefully did this game on my own. Initially, I was looking for a team but then I really wanted to challenge myself: can I build a game including assets, programming, level design, game design, writing, sound design and music myself in such a short time?

![challenge-accepted](/public/media/challenge-accepted-meme.jpg)

# Picking a theme

The theme has been announced around 6pm on Friday:

<blockquote class="twitter-tweet" data-lang="en" data-theme="dark"><p lang="de" dir="ltr">Der <a href="https://twitter.com/hashtag/miniBeansjam5?src=hash&amp;ref_src=twsrc%5Etfw">#miniBeansjam5</a> Countdown hat begonnen!<br>Hier sind die Begriffe für das <a href="https://twitter.com/hashtag/GameJam?src=hash&amp;ref_src=twsrc%5Etfw">#GameJam</a> Thema!<br><br>Wählt mindestens 2 der 3 Begriffe: Explosion, Elastisch und/oder Jenseits.<br><br>Die Zeit endet am Sonntag um 19:00 Uhr, also in 48 Stunden!<br>Viel Spaß bei unserem <a href="https://twitter.com/hashtag/GameJam?src=hash&amp;ref_src=twsrc%5Etfw">#GameJam</a>!<a href="https://twitter.com/hashtag/gamedev?src=hash&amp;ref_src=twsrc%5Etfw">#gamedev</a> <a href="https://twitter.com/hashtag/rbtv?src=hash&amp;ref_src=twsrc%5Etfw">#rbtv</a> <a href="https://t.co/5Rnw3iR1OM">pic.twitter.com/5Rnw3iR1OM</a></p>&mdash; miniBeansjam (@minibeansjam) <a href="https://twitter.com/minibeansjam/status/1246120117433520131?ref_src=twsrc%5Etfw">April 3, 2020</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>


 For my all non-German friends, this basically says to pick one or more of the following themes:

* Explosion
* Elastic
* Beyond

The themes are not super strict but can be interpreted in any way. Their main purpose is to guide your imagination! The first thing I always do is creating **a mindmap**:

![challenge-accepted](/public/media/mindmap-minibeansjam5.jpg)

You can use whatever tool you prefer for this, after a quick Google search I found [mindmapmaker.org](https://app.mindmapmaker.org/#m:new).

I stared on my creation for a couple of minutes until I realised: I want to build a **zombie shooter survival horror**!

# Setting up the project

The next step for me is to setup a project. Many people also do offline games (like boardgames) where this obviously is not applicable.

## Version control

A version control system (VSC) [like git](https://git-scm.com/) helps you to track your changes over time. The idea is that every change you do to your project is stored in the cloud and can be accessed at any given time. For that, I use Github for that:

[![shelter-github](/public/media/shelter-github-screenshot.jpg)](https://github.com/bitbrain/shelter)

It is important to use a VSC since it can save you a lot of pain. Using something like Dropbox is very difficult once you want to play around with features but you do not have the concept of branches. Also, working with multiple people on the same file can be challenging with something like Dropbox. Version control becomes a **MUST** in bigger teams.

## Generating the project files

Next up I generate the project. I build all my games in Java, more specifically with frameworks like [libgdx](https://libgdx.badlogicgames.com/) and [braingdx](https://github.com/bitbrain/braingdx). Then I commit the changes to Git and push them to my Github repository:

[![shelter-initial-commit](/public/media/shelter-initial-commit.jpg)](https://github.com/bitbrain/shelter/commit/778eadaff2618b342a05dcd64813310c0f482f9c)

Obviously, using when something like [Unity Engine](https://unity.com/), you'd push the project files of your Unity project.

# Plan your time wisely

This is the first critical stage. When I attended my first game jams a couple of years ago, I'd already rush ahead and implement **features** like health system, fighting, enemies or shooting. Those things are useful and make your game fun, however now might not be the right time to do this. In my head, it looks a little bit like this:

![48hour-timeline](/public/media/48-gamejam-timeline.jpg)

I call this the **Gamejam Flow Pyramid** (excuse my poor MSPaint skills. Also, arrows are now up to scale). The idea is the following:

* **Day 1**: working out the core mechanic of your game. After the first day, your game is basically already playable, but most likely is 0 fun since reward systems, UI, assets etc. are missing. However, **mechanically** the foundation for your game is set.
* **Day 2**: Powermode! It's time to implement all your features, assets and build a basic (functional) UI. At the end of the day, mostly only polishing and nice-to-have features should be left. Also, do not worry about the game loop or menu flow, that's all for the last day.
* **Day 3**: Wrapping it up. Time to finish the game loop, build menu flows (logo screen -> main menu -> ingame -> game over etc.) . Also try to polish your game as much as possible: add particle effects, screen shake, more animations etc. and add features you think will make the game more fun.

Obviously, this order is just a suggestion and always works for me the best. Also, for multi-disciplinary teams this order can be moved around or things like assets and levels can be prepared already on Day 1 while developers work on implementing game mechanics. On Day 2 and Day 3, artists, writers and composers can then work on polishing existing stuff or extend the game.

# Day 1: a man is (kinda) walking

TODO

# Day 2: zombie apocalypse

TODO

# Day 3: take shelter

TODO



