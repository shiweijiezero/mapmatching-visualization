<template>
  <div class="Home">
    <!--    <button v-on:click="showHelp">查看使用说明</button>-->
    <div id="myinput" style="float: left;width: 180px;">
      <div>
        <label>轨迹ID</label>
        <button v-on:click="randomVid">随机一个轨迹id</button>
        <!--        <input name="selectedVid" v-model="selectedVid">-->
        <input v-model="selectedVid"></input>
        <button v-on:click="genData">生成预处理数据</button>
      </div>

      <mu-container>
        <div>
          <label style="float: left">时间间隔阈值:{{interval}}</label>
          <mu-slider style="float: right" :step="1" :max="1000" class="demo-slider my-slider"
                     v-model="interval"></mu-slider>
        </div>
        <div>
          <label style="float: left">速度过滤阈值:{{speed}}</label>
          <mu-slider style="float: right" :step="1" :min=40 :max=150 class="demo-slider my-slider"
                     v-model="speed"></mu-slider>
        </div>
        <div>
          <label style="float: left">角度过滤阈值:{{angle}}</label>
          <mu-slider style="float: right" :step="1" :min=0 :max=180 class="demo-slider my-slider"
                     v-model="angle"></mu-slider>
        </div>
        <div>
          <label style="float: left">邻近路网绘制数量:{{roadmapK}}</label>
          <mu-slider style="float: right" :step="10" :min=0 :max=100 class="demo-slider my-slider"
                     v-model="roadmapK"></mu-slider>
        </div>
        <!--        <div>-->
        <!--          <label style="float: left">次序过滤窗口:{{orderwindow}}</label>-->
        <!--          <mu-slider style="float: right" :step="1" :min=0 :max=50 class="demo-slider"-->
        <!--                     v-model="orderwindow"></mu-slider>-->
        <!--        </div>-->
        <!--        <div>-->
        <!--          <label style="float: left">次序过滤步长:{{orderstep}}</label>-->
        <!--          <mu-slider style="float: right" :step="1" :min=0 :max=50 class="demo-slider" v-model="orderstep"></mu-slider>-->
        <!--        </div>-->
        <div>
          <label style="float: left">均值过滤窗口:{{windowsize}}</label>
          <mu-slider style="float: right" :step="2" :min=1 :max=100 class="demo-slider my-slider"
                     v-model="windowsize"></mu-slider>
        </div>
        <div>
          <label style="float: left">均值过滤sigmaM:{{sigmaM}}</label>
          <mu-slider style="float: right" :step="1" :min=10 :max=100 class="demo-slider my-slider"
                     v-model="sigmaM"></mu-slider>
        </div>
        <div>
          <label style="float: left">走廊半径宽度:{{corridor_width}}</label>
<!--          <mu-slider style="float: right" :step="1" :min=0 :max=20 class="demo-slider my-slider"-->
<!--                     v-model="corridor_width"></mu-slider>-->
          <input v-model="corridor_width"></input>
        </div>
        <div>
          <label style="float: left;">是否忽略稀疏点(1是,0否)</label>
          <input v-model="ignore_sparse"></input>
        </div>
        <div>
          <label style="float: left;">是否按照载客切分GPS轨迹</label>
          <input v-model="cutguest"></input>
        </div>
        <div>
          <label style="float: left">Java 匹配参数:</label>
          <input v-model="java_args"></input>
        </div>
        <div>
          <label style="float: left">python服务端口:</label>
          <input v-model="pyport"></input>
        </div>
<!--        <div>-->
<!--          <label style="float: left">java服务端口:</label>-->
<!--          <input v-model="javaport"></input>-->
<!--        </div>-->
      </mu-container>

      <div style="float: right">
        <div>
          <label>若未匹配过则耗时1分钟左右</label>
          <button v-on:click="doMatch">进行匹配</button>
        </div>
        <!--        <div>-->
        <!--          <label>选择匹配结果显示方案</label>-->
        <!--          <mu-radio style="margin-left: 16px" v-model="mrtype" name="showtype" value=0 label="路段"></mu-radio>-->
        <!--          <mu-radio style="margin-left: 16px" v-model="mrtype" name="showtype" value=1 label="路径"></mu-radio>-->
        <!--        </div>-->
<!--        <div>-->
<!--          <button v-on:click="clearData">清除[{{selectedVid}}]相关数据</button>-->
<!--        </div>-->
      </div>
    </div>

    <!--    <button v-on:click="doCheck">查看</button>-->
    <iframe class="renderTarget" v-on:load="iframe_onload" id="renderIframe" :src="renderUrl" style="float: left"
    ></iframe>
    <label :href="renderUrl">{{renderUrl}}</label>
    <audio id="alert_sound" src="http://data.huiyi8.com/2017/gha/03/17/1702.mp3" hidden="true"></audio>
  </div>
</template>

<script>

    function getRandomSubarray(arr, size) {
        var shuffled = arr.slice(0), i = arr.length, temp, index;
        while (i--) {
            index = Math.floor((i + 1) * Math.random());
            temp = shuffled[index];
            shuffled[index] = shuffled[i];
            shuffled[i] = temp;
        }
        return shuffled.slice(0, size);
    }

    import axios from 'axios';

    export default {
        props: {},
        data() {
            return {
                pyport: 5000,
                renderTarget: null,
                //backend_url: 'http://10.10.64.123:',
                // backend_url: 'http://192.168.126.193:',
                // backend_url: 'http://127.0.0.1:',
                backend_url: 'http://192.168.134.122:',
                //backend_url: 'http://10.10.65.143:',
                vids: null,
                selectedVid: '等待python服务启动...',
                renderUrl: null,
                interval: 500,
                speed: 130,
                angle: 100,
                roadmapK: 10,
                mrtype: 1,
                windowsize: 3,
                sigmaM: 50,
                corridor_width: 100,
                RMF: null,
                // java_args: '-mc20 -ms4|-mc100 -ms200 -tw0',
                java_args: '-mmOF-HMM -mc20 -sa3 -ms10 -tw110|-mmOF-CRF -mc65 -sa1 -ms200 -ce1 -dc40',
                orderwindow: 0,
                orderstep: 3,
                ignore_sparse: 0,
                javaport: 8090,
                cutguest: 0
            }
        },
        mounted() {
            this.getVids();
        },

        methods: {
            genData(event) {
                // document.getElementById("renderIframe").contentWindow.location.reload();
                this.renderUrl = this.backend_url + this.pyport + '/' + [
                    "render_processed",
                    this.cutguest,
                    this.javaport,
                    this.selectedVid,
                    this.interval,
                    this.speed,
                    this.angle,
                    this.roadmapK,
                    this.windowsize,
                    this.sigmaM,
                    this.corridor_width,
                    this.orderwindow,
                    this.orderstep,
                    this.ignore_sparse
                ].join('/');
            },
            getVids() {
                axios
                    .get(this.backend_url + this.pyport + '/' + 'getvids')
                    .then(response => {
                        // this.vids = getRandomSubarray(response.data, 50);
                      this.vids = response.data;
                        this.selectedVid = '245301_5';
                    })
            },
            doMatch() {
                // document.getElementById("renderIframe").contentWindow.location.reload(true);
                this.renderUrl = this.backend_url + this.pyport + '/' + ['domatch',
                    this.cutguest,
                    this.javaport,
                    this.mrtype,
                    this.selectedVid,
                    this.interval,
                    this.speed,
                    this.angle,
                    this.roadmapK,
                    this.windowsize,
                    this.sigmaM,
                    this.corridor_width,
                    this.java_args,
                    this.orderwindow,
                    this.orderstep,
                    this.ignore_sparse
                ].join('/');


            },
            clearData() {
                axios.get(this.backend_url + ['cleardata', this.selectedVid].join('/'))
                    .then(response => {
                        window.alert(response.data);
                    })
            },
            iframe_onload() {
                document.querySelector("#alert_sound").play();
            },
            // showHelp() {
            //     var helpmsg = "1.输入或选择车辆id\n" +
            //         "2.滑动滑块调整四个参数\n" +
            //         "3.生成并显示预处理数据\n" +
            //         "4.进行匹配并显示结果\n" +
            //         "5.如果想切换显示结果模式，则选择路段或者路径，再进行匹配\n" +
            //         "6.如果想要修改参数重新匹配，请清除相关数据，否则只会显示上一次匹配的结果";
            //     window.alert(helpmsg)
            // },
            randomVid() {
                this.selectedVid = getRandomSubarray(this.vids, 1)[0];
            },
            // renderComplete() {
            //     iframeTitle = document.getElementById("renderIframe").contentDocument.title;
            //     window.alert(iframeTitle);
            // }
        }
    };
</script>

<style>
  * {
    font-size: 13px;
    margin: 0px;
  }

  .my-slider {
    margin: 0px;
  }

  .selectVid {

  }

  .renderTarget_bak {
    width: 1280px;
    height: 675px;
  }
  .renderTarget {
    width: 1331px;
    height: 761px;
  }
</style>
