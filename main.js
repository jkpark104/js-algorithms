const input = 
`1 2
1 5
2 3
2 6
3 4
4 7
5 6
6 4`.split('\n')

const [v,e] = [7, 8]
let indegree = new Array(v+1).fill(0)
// let graph = new Array(v+1).fill([])
// 사용 X -> 참조 주소 들어감 
let graph = new Array(v+1).fill().map(() => [])
// let graph = Array.from({ length : v+1 }, () => [])

for (let i=0; i<e; i++) {
  const [a, b] = input[i].split(' ').map(x => parseInt(x))
  graph[a].push(b)
  indegree[b] += 1
}

topology_sort()

function topology_sort() {
  const q = []
  for (let i=1; i<v+1; i++) {
    if (indegree[i] === 0) {
      q.push(i)
    }
  }
  while (q.length !== 0) {
    const now = q.pop()

    console.log(now)

    for (next of graph[now]) {
      indegree[next] -= 1
      if (! indegree[next]) {
        q.push(next)
      }
    }
  }
}