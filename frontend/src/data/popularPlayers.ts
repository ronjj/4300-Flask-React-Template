import { PlayerCardData } from "../types";
import messiImg from "../assets/popular/messi.jpg";
import ronaldoImg from "../assets/popular/ronaldo.png";
import mbappeImg from "../assets/popular/mbappe.jpg";
import viniImg from "../assets/popular/vini.jpg";
import neymarImg from "../assets/popular/neymar.png";
import kaneImg from "../assets/popular/kane.png";
export const POPULAR_PLAYERS: PlayerCardData[] = [
  {
    key: "popular-messi",
    rank: 1,
    name: "Lionel Messi",
    team: "Fútbol Club Barcelona",
    position: "Forward",
    nationality: "Argentina",
    goals: 231,
    appearances: 243,
    image: messiImg,
  },
  {
    key: "popular-ronaldo",
    rank: 2,
    name: "Cristiano Ronaldo",
    team: "Al Nassr",
    position: "Forward",
    nationality: "Portugal",
    goals: 103,
    appearances: 141,
    image: ronaldoImg,
  },
  {
    key: "popular-mbappe",
    rank: 3,
    name: "Kylian Mbappé",
    team: "Real Madrid Club de Fútbol",
    position: "Forward",
    nationality: "France",
    goals: 54,
    appearances: 57,
    image: mbappeImg,
  },
  {
    key: "popular-vini",
    rank: 4,
    name: "Vinícius Júnior",
    team: "Real Madrid Club de Fútbol",
    position: "Forward",
    nationality: "Brazil",
    goals: 70,
    appearances: 233,
    image: viniImg,
  },
  {
    key: "popular-neymar",
    rank: 5,
    name: "Neymar",
    team: "Fútbol Club Barcelona",
    position: "Forward",
    nationality: "Brazil",
    goals: 59,
    appearances: 97,
    image: neymarImg,
  },
  {
    key: "popular-kane",
    rank: 6,
    name: "Harry Kane",
    team: "FC Bayern München",
    position: "Forward",
    nationality: "England",
    goals: 213,
    appearances: 320,
    image: kaneImg,
  },
];
export default POPULAR_PLAYERS;