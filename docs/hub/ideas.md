---
title: Ideas and vault
order: 9
summary: Banked ideas, stretch goals and later phases, kept out of the active backlog.
---
# Ideas and vault

Parked ideas live here, not in [decisions](decisions.md) or [backlog](backlog.md). Dan promotes an idea to the backlog; nobody else does.

| Idea | Why | Effort | Depends on | Phase |
|---|---|---|---|---|
| Local agents that build a game through the app ([FU-126](tracker/open.md)) | One local app, one memory manager, one API for code, art, voice and video | M (thin MCP layer, 1 to 2 days) plus a local coding model that is good enough | A local model close to a hosted frontier one | Banked |
| Fit badge and recommended settings per hardware ([FU-133](tracker/open.md)) | Tells the user what runs well on their machine and which Studio settings to turn on | M to L (1.5 days badge, 2 days settings with measurements) | The shared model card from the Models page merge; measurements on 3 to 4 families | Later |
| One Models page per area instead of Discover and My Models ([FU-129](tracker/open.md)) | The screens are nearly identical | L (2 to 4 days) | After v0.9.9 | Later |
| Image-to-video and sound-to-video on the LTX engine | The engine supports both; the request has no image or audio field yet | M | Video request schema change | Later |
| oMLX as an engine lane ([FU-132](tracker/open.md)) | SSD-persistent KV cache and concurrent requests | M | A measured win over our MLX lane | Later |
| LongLive real-time long video on CUDA ([FU-003](tracker/open.md)) | Long causal clips | L | A CUDA dev box | Later |
| KVTC cache strategy ([FU-029](tracker/open.md)) | 8 to 32 times KV compression | L | MLX or Metal port upstream | Later |
| Dialogue scripts and voice cloning ([FU-117](tracker/open.md)) | Voice studio completeness | M | Licence and speed check | Later |
| More mflux families (Z-Image-Turbo, Krea 2, MOSS-Image, Ming-Image) ([FU-096](tracker/open.md), [FU-123](tracker/open.md)) | Newer and faster image models | S each | A table row plus a catalog row | Later |

## Later phases

- **After v0.9.9:** Models page consolidation; Windows and Linux validation sweep.
- **Watch list** (no runnable weights or runtime yet): Qwen 3.7, Wan 2.6 / 2.7 / 3.0, MiniMax-H3, LongCat-Video ([FU-082](tracker/open.md)).
