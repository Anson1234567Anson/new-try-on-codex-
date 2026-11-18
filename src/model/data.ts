// The course's running example: a tiny model that learns to write first names.

export const NAMES = `
emma olivia ava sophia mia amelia harper evelyn abigail emily ella scarlett grace chloe
victoria riley aria lily aubrey zoey penelope layla nora camila hannah lillian addison
eleanor natalie luna savanna brookly leah zoe stella hazel ellie paisley audrey skylar
violet claire bella aurora lucy anna samanth carolin genesis aaliyah kennedy kinsley
allison maya sarah madelyn adeline alexa ariana elena gabriel naomi alice sadie hailey
eva emilia autumn quinn nevaeh piper ruby serenit willow everly cora kaylee lydia
aubree arianna eliana peyton melanie gianna isabell julia valenti nova clara vivian
reagan mackenz madelin brielle delilah isla rylee katheri sophie josephi ivy liliana
jade maria taylor hadley kylie emery adalynn natalia annabel faith alexand ximena
ashley brianna raelynn bailey mary athena andrea leilani jasmine lyla margare alyssa
adalyn arya norah khloe kayla eden eliza rose ariel melody alexis isabel sydney
juliana lauren iris emerson london morgan lilly charlie aliyah valeria arabell sara
finley trinity ryleigh jordyn jocelyn kimberl esther molly valerie cecilia anastas
daisy reese laila mya amy teagan amaya elise harmony paige adaline fiona alaina
nicole genevie lucia alina mckenzi callie payton eloise brookes londyn mariah julian
liam noah william james oliver benjami elijah lucas mason logan alexand ethan jacob
michael daniel henry jackson sebasti aiden matthew samuel david joseph carter owen
wyatt john jack luke jayden dylan grayson levi isaac gabriel julian mateo anthony
jaxon lincoln joshua christo andrew theodor caleb ryan asher nathan thomas leo isaiah
charles josiah hudson christi hunter connor eli ezra aaron landon adrian jonatha nolan
jeremia easton elias colton cameron carson robert angel maveric nichola dominic
jaxson greyson adam ian austin santiag jordan cooper brayden roman evan ezekiel
xavier jose jace jameson leonard bryson axel everett parker kayden miles sawyer jason
declan weston micah ayden wesley luca vincent damian zachary silas gavin chase kai
emmett harriso nathani kingsto cole tyler bennett bentley ryker tristan brandon kevin
luis george ashton rowan braxton ryder gael ivan diego maxwell max carlos kaiden
juan maddox justin waylon calvin giovann jonah abel jayce jesus amir king beau
tucker emilian marcus rhett brody dean jude nicolas ryland blake rory felix simon
arthur louis oscar hugo otto milo enzo nico remy theo finn dante elliot peter paul
mark tim tom ben sam dan ray lee joe amy ann eve kim liz meg sue ada ida ina una
`.trim().split(/\s+/).filter((n, i, all) => n.length <= 7 && all.indexOf(n) === i);

/** Vocabulary: '.' marks the start and the end of a name, then a–z. */
export const CHARS = '.abcdefghijklmnopqrstuvwxyz';
export const VOCAB = CHARS.length;
export const CTX = 8;

export const encode = (s: string): number[] => [...s].map((ch) => CHARS.indexOf(ch));
export const decode = (ids: number[]): string => ids.map((i) => CHARS[i]).join('');

/** One training example: the model reads `x` and should predict `y`, one step ahead. */
export function example(name: string): { x: number[]; y: number[] } {
  const seq = encode('.' + name + '.');
  return { x: seq.slice(0, -1).slice(0, CTX), y: seq.slice(1).slice(0, CTX) };
}

/** Small seeded random number generator, so every run is repeatable. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rand: () => number): number {
  return Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
}
