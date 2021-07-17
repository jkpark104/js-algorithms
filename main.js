// const input = 
// `1 2 29
// 1 5 75
// 2 3 35
// 2 6 34
// 3 4 7
// 4 6 23
// 4 7 13
// 5 6 53
// 6 7 25`.split('\n')

// const [v, e] = [7, 9]
// const edges = []

// for (el of input) {
//   edges.push(el.split(' ').map(x => parseInt(x)))
// }

// edges.sort((a,b) => console.log(a,b,a[2]-b[2]))
// // console.log(edges)

// function find_parent(x) {
//   if (x != parent[x]) {
//     parent[x] = find_parent(parent[x])
//   }
//   return parent[x]
// }

// function union_parent(a, b) {
//   a = find_parent(a)
//   b = find_parent(b)
//   if (a > b) {
//     parent[a] = b
//   } else {
//     parent[b] = a
//   }
// }

let array = [[9,1], [7,3], [12,4], [3,9]]

array.sort((a,b) => b[1]-a[1])

console.log(array)